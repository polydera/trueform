/*
 * Copyright (c) 2025 XLAB
 * All rights reserved.
 *
 * This file is part of trueform (trueform.polydera.com)
 *
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 *
 * Author: Žiga Sajovic
 */
#pragma once

#include "../core/buffer.hpp"
#include "../core/constants.hpp"
#include "../core/edges.hpp"
#include "../core/points.hpp"
#include "../core/views/constant.hpp"
#include "../core/views/offset_block_range.hpp"
#include "../exact/resolve_int_type.hpp"
#include "./cdt_refine_config.hpp"
#include "./detail/size_adaptive.hpp"
#include "./constrained_delaunay_triangulator.hpp"
#include <tbb/parallel_sort.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

namespace tf {

/// @ingroup topology
/// @brief One recorded constraint split: the input constraint edge and the
/// dyadic parameter numerator/2^depth along it (numerator odd, canonical).
template <typename Index> struct cdt_constraint_split {
  Index edge;
  std::uint8_t numerator;
  std::uint8_t depth;
};

/// @ingroup topology
/// @brief Incremental Delaunay refinement over the constraint-preserving CDT.
///
/// The boundary CDT is built once (constraints never intersected) and adopted
/// into a persistent triangle structure; a bad-triangle queue then drives
/// circumcenter insertions directly -- no rebuild rounds and no point
/// location, since every candidate walks from its generating triangle.
/// Triangles are judged by the quality measure q = (2/sqrt3) * 2A /
/// max_edge^2. Constrained edges are never flipped; a candidate that
/// encroaches a constraint's diametral circle splits the constraint at its
/// midpoint instead (Ruppert), as does an existing vertex inside a diametral
/// circle. Vertices constraint-connected to a segment's endpoint are exempt
/// encroachers (sharp input corners would otherwise split forever), and each
/// input constraint splits at most k_max_split_depth generations.
///
/// Every split is recorded as a dyadic parameter of the ORIGINAL input edge
/// (see @ref cdt_constraint_split), so callers coordinating several
/// triangulations over shared polylines can union the records and reproduce
/// identical split points. Steiner positions are deterministic for identical
/// input.
///
/// Build returns false (bail, result unusable) on input the preserve contract
/// rejects: crossing constraints, a point in a constraint's interior, or a
/// degenerate constraint.
///
/// @tparam Index The index type.
/// @tparam Coord The input coordinate type; integral coordinates are consumed
///   exactly (identity conversion). To make dyadic split positions exact on
///   an integer lattice, reserve k_max_split_depth bits of scale headroom.
/// @tparam Int The integer type for exact predicates; defaults per Coord
///   via @ref tf::exact::resolve_int_type (int32 for float, int64 for
///   double, identity for integral Coord).
template <typename Index, typename Coord,
          typename Int = tf::exact::resolve_int_type<tf::none_t, Coord>>
class cdt_refiner {
  static constexpr int k_max_split_depth = 6;
  static constexpr Index k_none = Index(-1);

public:
  auto clear() -> void {
    _cdt.clear();
    _ip.clear();
    _dp.clear();
    _t.clear();
    _label.clear();
    _segments.clear();
    _con_nbr.clear();
    _queue.clear();
    _cavity.clear();
    _lawson.clear();
    _pending.clear();
    _generation.clear();
    _splits.clear();
    _split_offsets.clear();
    _min_quality = 0.0;
    _scale = 1;
    _offx = 0;
    _offy = 0;
    _n_input_points = 0;
    _n_input_edges = 0;
    _ok = false;
  }

  template <typename PointsPolicy, typename EdgesPolicy,
            typename IsBoundaryRange>
  auto build(const tf::points<PointsPolicy> &pts,
             const tf::edges<EdgesPolicy> &edges,
             const IsBoundaryRange &is_boundary,
             const tf::cdt_refine_config &config = {}) -> bool {
    return build(pts, edges, is_boundary,
                 tf::make_constant_range(true, edges.size()), config);
  }

  /// @brief Build with a per-edge `is_splittable` mask: a non-splittable
  /// constraint never receives midpoint splits (it is born at the split
  /// depth cap), regardless of config.split_encroached.
  template <typename PointsPolicy, typename EdgesPolicy,
            typename IsBoundaryRange, typename IsSplittableRange>
  auto build(const tf::points<PointsPolicy> &pts,
             const tf::edges<EdgesPolicy> &edges,
             const IsBoundaryRange &is_boundary,
             const IsSplittableRange &is_splittable,
             const tf::cdt_refine_config &config) -> bool {
    clear();
    _n_input_points = static_cast<Index>(pts.size());
    _n_input_edges = static_cast<Index>(edges.size());
    _split_encroached = config.split_encroached;

    if (!_cdt.build(pts, edges, is_boundary, /*split_constraints=*/false))
      return _ok = false;

    adopt(edges, is_splittable);
    // circumcenter refinement is not guaranteed to terminate above 0.45
    refine(std::clamp(double(config.min_quality), 0.0, 0.45));
    finalize_splits();
    return _ok = true;
  }

  template <typename PointsPolicy, typename EdgesPolicy>
  auto build(const tf::points<PointsPolicy> &pts,
             const tf::edges<EdgesPolicy> &edges,
             const tf::cdt_refine_config &config = {}) -> bool {
    return build(pts, edges, tf::make_constant_range(true, edges.size()),
                 config);
  }

  auto ok() const -> bool { return _ok; }
  auto n_input_points() const -> Index { return _n_input_points; }

  /// @brief Final triangulation faces (all; filter by region_labels parity).
  /// @brief Number of output faces; face ids align with region_labels.
  auto n_faces() const -> Index { return static_cast<Index>(_t.size()); }

  /// @brief Face corners by face id (no allocation; region_labels order).
  auto face(Index i) const -> std::array<Index, 3> {
    const auto &t = _t[std::size_t(i)];
    return {t.v[0], t.v[1], t.v[2]};
  }

  auto make_faces() const -> tf::blocked_buffer<Index, 3> {
    tf::blocked_buffer<Index, 3> out;
    out.reserve(_t.size());
    for (const auto &f : _t)
      out.push_back({f.v[0], f.v[1], f.v[2]});
    return out;
  }

  /// @brief Per-face region parity labels (match make_faces order).
  auto region_labels() const { return tf::make_range(_label); }

  /// @brief Output points on the internal integer lattice.
  auto points() const { return tf::make_points(_ip); }

  /// @brief Output points deconverted to Coord.
  auto converted_points() const {
    if constexpr (std::is_integral_v<Coord>) {
      return tf::make_points(tf::make_mapped_range(_ip, [](const auto &p) {
        return tf::point<Coord, 2>{Coord(p[0]), Coord(p[1])};
      }));
    } else {
      return tf::make_points(tf::make_mapped_range(_dp, [](const auto &p) {
        return tf::point<Coord, 2>{Coord(p[0]), Coord(p[1])};
      }));
    }
  }

  /// @brief Index map between input point space and output point space.
  auto index_map() const -> const tf::index_map_buffer<Index> & {
    return _cdt.index_map();
  }

  /// @brief Per input constraint edge, its recorded splits sorted by
  /// parameter. Indexed by the input edge id.
  auto constraint_splits() const {
    return tf::make_offset_block_range(_split_offsets, _splits);
  }
  auto n_constraint_splits() const -> Index {
    return static_cast<Index>(_splits.size());
  }

/// @brief Is edge `e` of output face `f` (make_faces order) constrained?
  auto constrained(Index f, int e) const -> bool {
    return _t[f].seg[e] != k_none;
  }

private:
  struct triangle {
    Index v[3];
    Index n[3];   // neighbor across edge v[e] -> v[e+1]
    Index seg[3]; // constrained sub-edge id, k_none when unconstrained
    std::uint32_t stamp;
  };
  // diametral-circle encroachment: p sees the segment (u, v) at an obtuse
  // angle (Ruppert)
  auto encroaches(const std::array<double, 2> &pu,
                  const std::array<double, 2> &pv,
                  const std::array<double, 2> &pp) const -> bool {
    double ux = pp[0] - pu[0], uy = pp[1] - pu[1];
    double vx = pp[0] - pv[0], vy = pp[1] - pv[1];
    return ux * vx + uy * vy < 0;
  }
  // A constrained sub-edge: interval [m/2^d, (m+1)/2^d] of input edge
  // `origin`, with `lo_vertex` the endpoint on the m/2^d side.
  struct sub_edge {
    Index origin;
    Index lo_vertex;
    std::uint8_t m;
    std::uint8_t d;
  };

  auto orient(Index a, Index b, Index c) const -> int {
    return tf::exact::orient2d_sign(_ip[a], _ip[b], _ip[c]);
  }
  auto incircle(Index a, Index b, Index c, Index d) const -> int {
    return tf::exact::incircle_sign(_ip[a], _ip[b], _ip[c], _ip[d]);
  }

  auto quality(Index f) const -> double {
    const auto &a = _dp[std::size_t(_t[f].v[0])];
    const auto &b = _dp[std::size_t(_t[f].v[1])];
    const auto &c = _dp[std::size_t(_t[f].v[2])];
    double ex = b[0] - a[0], ey = b[1] - a[1];
    double fx = c[0] - a[0], fy = c[1] - a[1];
    double area2 = std::abs(ex * fy - ey * fx);
    double e2 = ex * ex + ey * ey;
    double f2 = fx * fx + fy * fy;
    double g2 = (fx - ex) * (fx - ex) + (fy - ey) * (fy - ey);
    double m = std::max({e2, f2, g2});
    return m > 0 ? tf::two_over_sqrt_3<double> * area2 / m : 1.0;
  }

  auto constraint_connected(Index a, Index b) const -> bool {
    return _con_nbr[std::size_t(a)][0] == b ||
           _con_nbr[std::size_t(a)][1] == b ||
           _con_nbr[std::size_t(b)][0] == a || _con_nbr[std::size_t(b)][1] == a;
  }

  auto edge_back(Index f, Index from) const -> int {
    for (int e = 0; e < 3; ++e)
      if (_t[f].n[e] == from)
        return e;
    return -1;
  }

  auto sub_depth(Index f, int e) const -> int {
    auto s = _t[f].seg[e];
    return s == k_none ? 0 : int(_segments[std::size_t(s)].d);
  }
  auto splittable(Index f, int e) const -> bool {
    return _split_encroached && sub_depth(f, e) < k_max_split_depth;
  }

  auto push_bad(Index f) -> void {
    if (_t[f].seg[0] == k_none && _t[f].seg[1] == k_none &&
        _t[f].seg[2] == k_none && quality(f) >= _min_quality)
      return;
    _queue.push_back({f, Index(_t[f].stamp)});
  }
  auto push_lawson(Index f, int e) -> void {
    _lawson.push_back((f << 2) | e);
  }

  // Lawson restoration from the recorded suspect edges; constrained edges are
  // never flipped -- if the fresh vertex encroaches one, its split is
  // requested instead.
  auto restore(Index fresh) -> void {
    while (_lawson.size() != 0) {
      Index packed = _lawson.back();
      _lawson.pop_back();
      Index f = packed >> 2;
      int e = packed & 3;
      Index g = _t[f].n[e];
      Index a = _t[f].v[e];
      Index b = _t[f].v[(e + 1) % 3];
      Index c = _t[f].v[(e + 2) % 3];
      if (constrained(f, e)) {
        if (fresh != a && fresh != b && splittable(f, e) &&
            !constraint_connected(fresh, a) && !constraint_connected(fresh, b)) {
          if (encroaches(_dp[std::size_t(a)], _dp[std::size_t(b)],
                         _dp[std::size_t(fresh)]))
            _pending.push_back((f << 2) | e);
        }
        continue;
      }
      if (g == k_none)
        continue;
      int eb = edge_back(g, f);
      Index d = _t[g].v[(eb + 2) % 3];
      if (c == d)
        continue;
      if (incircle(a, b, c, d) <= 0)
        continue;
      if (orient(a, d, c) <= 0 || orient(b, c, d) <= 0)
        continue; // flip would create a degenerate or inverted triangle

      // flip f/g: shared edge a-b becomes c-d
      Index nb_bc = _t[f].n[(e + 1) % 3];
      Index nb_ca = _t[f].n[(e + 2) % 3];
      Index nb_ad = _t[g].n[(eb + 1) % 3];
      Index nb_db = _t[g].n[(eb + 2) % 3];
      Index s_bc = _t[f].seg[(e + 1) % 3];
      Index s_ca = _t[f].seg[(e + 2) % 3];
      Index s_ad = _t[g].seg[(eb + 1) % 3];
      Index s_db = _t[g].seg[(eb + 2) % 3];

      _t[f].v[0] = a;
      _t[f].v[1] = d;
      _t[f].v[2] = c;
      _t[f].n[0] = nb_ad;
      _t[f].n[1] = g;
      _t[f].n[2] = nb_ca;
      _t[f].seg[0] = s_ad;
      _t[f].seg[1] = k_none;
      _t[f].seg[2] = s_ca;
      _t[g].v[0] = b;
      _t[g].v[1] = c;
      _t[g].v[2] = d;
      _t[g].n[0] = nb_bc;
      _t[g].n[1] = f;
      _t[g].n[2] = nb_db;
      _t[g].seg[0] = s_bc;
      _t[g].seg[1] = k_none;
      _t[g].seg[2] = s_db;
      if (nb_ad != k_none) {
        int k = edge_back(nb_ad, g);
        if (k >= 0)
          _t[nb_ad].n[k] = f;
      }
      if (nb_bc != k_none) {
        int k = edge_back(nb_bc, f);
        if (k >= 0)
          _t[nb_bc].n[k] = g;
      }
      ++_t[f].stamp;
      ++_t[g].stamp;
      push_lawson(f, 0);
      push_lawson(f, 2);
      push_lawson(g, 0);
      push_lawson(g, 2);
      push_bad(f);
      push_bad(g);
    }
  }

  // 1->3 split of triangle f at interior vertex p
  auto split_triangle(Index f, Index p) -> void {
    Index v0 = _t[f].v[0], v1 = _t[f].v[1], v2 = _t[f].v[2];
    Index n0 = _t[f].n[0], n1 = _t[f].n[1], n2 = _t[f].n[2];
    Index s0 = _t[f].seg[0], s1 = _t[f].seg[1], s2 = _t[f].seg[2];
    Index lab = _label[std::size_t(f)];
    Index ta = Index(_t.size());
    Index tb = ta + 1;
    std::uint32_t stamp = _t[f].stamp + 1;
    _t[f] = {{v0, v1, p}, {n0, ta, tb}, {s0, k_none, k_none}, stamp};
    _t.push_back({{v1, v2, p}, {n1, tb, f}, {s1, k_none, k_none}, 0});
    _t.push_back({{v2, v0, p}, {n2, f, ta}, {s2, k_none, k_none}, 0});
    _label.push_back(lab);
    _label.push_back(lab);
    if (n1 != k_none) {
      int k = edge_back(n1, f);
      if (k >= 0)
        _t[n1].n[k] = ta;
    }
    if (n2 != k_none) {
      int k = edge_back(n2, f);
      if (k >= 0)
        _t[n2].n[k] = tb;
    }
    push_lawson(f, 0);
    push_lawson(ta, 0);
    push_lawson(tb, 0);
    push_bad(f);
    push_bad(ta);
    push_bad(tb);
    restore(p);
  }

  // 2->4 split of edge e of triangle f at vertex p (on the edge); when the
  // edge is a constrained sub-edge, s_lo/s_hi are the child sub-edge ids
  auto split_edge(Index f, int e, Index p, Index s_lo, Index s_hi) -> void {
    Index g = _t[f].n[e];
    Index a = _t[f].v[e];
    Index b = _t[f].v[(e + 1) % 3];
    Index c = _t[f].v[(e + 2) % 3];
    Index nb_bc = _t[f].n[(e + 1) % 3];
    Index nb_ca = _t[f].n[(e + 2) % 3];
    Index s_bc = _t[f].seg[(e + 1) % 3];
    Index s_ca = _t[f].seg[(e + 2) % 3];
    Index lab_f = _label[std::size_t(f)];
    // child sub-edge ids by which endpoint sits on the lo side
    Index s_ap = s_lo, s_pb = s_hi;
    if (s_lo != k_none && _segments[std::size_t(s_lo)].lo_vertex != a)
      std::swap(s_ap, s_pb);

    Index fa = Index(_t.size());
    std::uint32_t stamp = _t[f].stamp + 1;
    // f: (a, p, c), fa: (p, b, c)
    _t[f] = {{a, p, c}, {k_none, fa, nb_ca}, {s_ap, k_none, s_ca}, stamp};
    _t.push_back({{p, b, c}, {k_none, nb_bc, f}, {s_pb, s_bc, k_none}, 0});
    _label.push_back(lab_f);
    if (nb_bc != k_none) {
      int k = edge_back(nb_bc, f);
      if (k >= 0)
        _t[nb_bc].n[k] = fa;
    }
    _t[f].n[1] = fa;
    _t[fa].n[2] = f;

    if (g != k_none) {
      int eb = edge_back(g, f);
      // g's edges around the old (b, a) orientation: eb = b->a,
      // eb+1 = a->d, eb+2 = d->b
      Index d = _t[g].v[(eb + 2) % 3];
      Index nb_ad = _t[g].n[(eb + 1) % 3];
      Index nb_db = _t[g].n[(eb + 2) % 3];
      Index s_ad = _t[g].seg[(eb + 1) % 3];
      Index s_db = _t[g].seg[(eb + 2) % 3];
      Index lab_g = _label[std::size_t(g)];
      Index ga = Index(_t.size());
      std::uint32_t gstamp = _t[g].stamp + 1;
      // g: (b, p, d), ga: (p, a, d)
      _t[g] = {{b, p, d}, {k_none, ga, nb_db}, {s_pb, k_none, s_db}, gstamp};
      _t.push_back(
          {{p, a, d}, {k_none, nb_ad, g}, {s_ap, s_ad, k_none}, 0});
      _label.push_back(lab_g);
      if (nb_ad != k_none) {
        int k = edge_back(nb_ad, g);
        if (k >= 0)
          _t[nb_ad].n[k] = ga;
      }
      // stitch across the split edge
      _t[f].n[0] = ga;
      _t[ga].n[0] = f;
      _t[fa].n[0] = g;
      _t[g].n[0] = fa;
      push_lawson(g, 2);
      push_lawson(ga, 1);
      push_bad(g);
      push_bad(ga);
    }
    push_lawson(f, 2);
    push_lawson(fa, 1);
    push_bad(f);
    push_bad(fa);
    restore(p);
  }

  // walk from `from` toward point p (already appended); returns
  // (triangle, -1) when inside, (triangle, edge) when blocked by a
  // constrained edge, (k_none, -1) when out of the region
  auto walk(Index from, Index p) const -> std::pair<Index, int> {
    Index f = from;
    for (int guard = 0; guard < 1024; ++guard) {
      int exit_edge = -1;
      for (int e = 0; e < 3; ++e)
        if (orient(_t[f].v[e], _t[f].v[(e + 1) % 3], p) < 0) {
          exit_edge = e;
          break;
        }
      if (exit_edge < 0)
        return {f, -1};
      if (constrained(f, exit_edge))
        return {f, exit_edge};
      Index g = _t[f].n[exit_edge];
      if (g == k_none)
        return {k_none, -1};
      f = g;
    }
    return {k_none, -1};
  }

  // Ruppert rule 1: a constrained edge whose diametral circle contains the
  // apex of its triangle is encroached and must be split (sharp input
  // corners exempt, depth capped)
  auto queue_encroached(Index f) -> bool {
    bool any = false;
    for (int e = 0; e < 3; ++e) {
      if (!constrained(f, e) || !splittable(f, e))
        continue;
      Index u = _t[f].v[e];
      Index v = _t[f].v[(e + 1) % 3];
      Index w = _t[f].v[(e + 2) % 3];
      if (constraint_connected(w, u) || constraint_connected(w, v))
        continue;
      if (encroaches(_dp[std::size_t(u)], _dp[std::size_t(v)],
                     _dp[std::size_t(w)])) {
        _pending.push_back((f << 2) | e);
        any = true;
      }
    }
    return any;
  }

  // perform exactly one generation of segment splits; nested requests
  // produced by the split restores only requeue their triangle (the
  // encroachment is rediscovered at pop time)
  auto drain_one_generation() -> void {
    _generation.clear();
    std::swap(_generation, _pending);
    for (Index packed : _generation) {
      Index sf = packed >> 2;
      int se = packed & 3;
      if (!constrained(sf, se))
        continue;
      Index u = _t[sf].v[se];
      Index v = _t[sf].v[(se + 1) % 3];
      Index m = Index(_ip.size());
      _ip.push_back(tf::point<Int, 2>{
          Int((std::int64_t(_ip[u][0]) + _ip[v][0]) / 2),
          Int((std::int64_t(_ip[u][1]) + _ip[v][1]) / 2)});
      if ((_ip[m][0] == _ip[u][0] && _ip[m][1] == _ip[u][1]) ||
          (_ip[m][0] == _ip[v][0] && _ip[m][1] == _ip[v][1])) {
        _ip.pop_back();
        continue; // constraint below lattice resolution
      }
      _dp.push_back({(double(_ip[m][0]) - _offx) / _scale,
                     (double(_ip[m][1]) - _offy) / _scale});
      Index s = _t[sf].seg[se];
      Index s_lo = k_none, s_hi = k_none;
      if (s != k_none) {
        auto seg = _segments[std::size_t(s)];
        _splits.push_back({seg.origin, std::uint8_t(2 * seg.m + 1),
                           std::uint8_t(seg.d + 1)});
        s_lo = Index(_segments.size());
        s_hi = s_lo + 1;
        _segments.push_back({seg.origin, seg.lo_vertex,
                             std::uint8_t(2 * seg.m), std::uint8_t(seg.d + 1)});
        _segments.push_back(
            {seg.origin, m, std::uint8_t(2 * seg.m + 1),
             std::uint8_t(seg.d + 1)});
      }
      _con_nbr.push_back({u, v});
      auto relink = [&](Index x, Index from, Index to) {
        if (_con_nbr[std::size_t(x)][0] == from)
          _con_nbr[std::size_t(x)][0] = to;
        else if (_con_nbr[std::size_t(x)][1] == from)
          _con_nbr[std::size_t(x)][1] = to;
      };
      relink(u, v, m);
      relink(v, u, m);
      split_edge(sf, se, m, s_lo, s_hi);
    }
    for (Index packed : _pending)
      push_bad(packed >> 2);
    _pending.clear();
  }

  auto refine(double min_quality) -> void {
    // popping a face whose quality already clears the floor and which has
    // no constrained-splittable edge is a pure no-op, so such faces are
    // never queued; with a zero floor and segment splits off, that is
    // every face
    if (min_quality <= 0.0 && !_split_encroached)
      return;
    // in 64-bit: 64 * n_points overflows a 32-bit Index above ~2^25 inputs
    const std::int64_t point_budget =
        std::int64_t(64) * std::int64_t(_ip.size()) + 4096;
    _min_quality = min_quality;
    for (Index f = 0; f < Index(_t.size()); ++f)
      if (_label[std::size_t(f)] % 2 == 1)
        push_bad(f);

    std::size_t head = 0;
    while (head < _queue.size()) {
      if (std::int64_t(_ip.size()) >= point_budget)
        return;
      auto [f, stamp] = _queue[head++];
      if (head > (std::size_t(1) << 16) && head * 2 > _queue.size()) {
        _queue.erase(_queue.begin(), _queue.begin() + head);
        head = 0;
      }
      if (std::uint32_t(stamp) != _t[f].stamp)
        continue;
      if (queue_encroached(f)) {
        drain_one_generation();
        continue; // requeued by the split ops themselves
      }
      if (quality(f) >= min_quality)
        continue;

      const auto &a = _dp[std::size_t(_t[f].v[0])];
      const auto &b = _dp[std::size_t(_t[f].v[1])];
      const auto &c = _dp[std::size_t(_t[f].v[2])];
      double ex = b[0] - a[0], ey = b[1] - a[1];
      double fx = c[0] - a[0], fy = c[1] - a[1];
      double d = 2.0 * (ex * fy - ey * fx);
      if (d == 0)
        continue;
      double le2 = ex * ex + ey * ey, lf2 = fx * fx + fy * fy;
      double ccx = a[0] + (fy * le2 - ey * lf2) / d;
      double ccy = a[1] + (ex * lf2 - fx * le2) / d;
      double gx = ccx * _scale + _offx;
      double gy = ccy * _scale + _offy;
      const double lattice_max =
          double(std::numeric_limits<Int>::max()) * 0.5;
      if (!(std::abs(gx) < lattice_max) || !(std::abs(gy) < lattice_max))
        continue; // far outside the lattice: unreachable anyway

      Index p = Index(_ip.size());
      _ip.push_back(
          tf::point<Int, 2>{Int(std::llround(gx)), Int(std::llround(gy))});
      // double positions must be the deconversion of the int positions, or
      // double-space geometry diverges from the exact topology at sliver
      // scales
      _dp.push_back({(double(_ip[p][0]) - _offx) / _scale,
                     (double(_ip[p][1]) - _offy) / _scale});

      auto [host, blocked] = walk(f, p);
      if (host == k_none) {
        _ip.pop_back();
        _dp.pop_back();
        continue;
      }
      if (blocked >= 0) {
        _ip.pop_back();
        _dp.pop_back();
        Index bu = _t[host].v[blocked];
        Index bv = _t[host].v[(blocked + 1) % 3];
        if (encroaches(_dp[std::size_t(bu)], _dp[std::size_t(bv)],
                       {ccx, ccy}) &&
            splittable(host, blocked))
          _pending.push_back((host << 2) | blocked);
      } else {
        // Ruppert: the candidate must not encroach any constrained edge of
        // its cavity -- split the segment instead (or drop when capped)
        _cavity.clear();
        _cavity.push_back(host);
        bool encroaches_capped = false;
        Index enc_split = k_none;
        for (std::size_t ci = 0; ci < _cavity.size() && ci < 64; ++ci) {
          Index cf = _cavity[ci];
          for (int e = 0; e < 3; ++e) {
            Index cu = _t[cf].v[e];
            Index cv = _t[cf].v[(e + 1) % 3];
            if (constrained(cf, e)) {
              if (encroaches(_dp[std::size_t(cu)], _dp[std::size_t(cv)],
                             _dp[std::size_t(p)])) {
                if (splittable(cf, e))
                  enc_split = (cf << 2) | e;
                else
                  encroaches_capped = true;
              }
              continue;
            }
            Index cg = _t[cf].n[e];
            if (cg == k_none)
              continue;
            bool seen = false;
            for (Index x : _cavity)
              if (x == cg) {
                seen = true;
                break;
              }
            if (seen)
              continue;
            int eb = edge_back(cg, cf);
            Index apex = _t[cg].v[(eb + 2) % 3];
            if (incircle(cu, cv, _t[cf].v[(e + 2) % 3], apex) >= 0 ||
                incircle(_t[cg].v[eb], _t[cg].v[(eb + 1) % 3], apex, p) > 0)
              _cavity.push_back(cg);
          }
        }
        if (encroaches_capped || enc_split != k_none) {
          _ip.pop_back();
          _dp.pop_back();
          if (enc_split != k_none) {
            _pending.push_back(enc_split);
            push_bad(f); // the split changes the neighbourhood: retry
            drain_one_generation();
          }
          // capped: input-limited, no retry, no spin
          continue;
        }
        bool dup = false;
        for (int k = 0; k < 3; ++k)
          if (_ip[_t[host].v[k]][0] == _ip[p][0] &&
              _ip[_t[host].v[k]][1] == _ip[p][1])
            dup = true;
        if (dup) {
          _ip.pop_back();
          _dp.pop_back();
          continue;
        }
        // quantization can drop p exactly onto an edge of the host: a 1->3
        // split would create a degenerate child, so split the edge instead
        int on_edge = -1;
        for (int e = 0; e < 3; ++e)
          if (orient(_t[host].v[e], _t[host].v[(e + 1) % 3], p) == 0) {
            on_edge = e;
            break;
          }
        if (on_edge >= 0) {
          if (constrained(host, on_edge) || _t[host].n[on_edge] == k_none) {
            _ip.pop_back();
            _dp.pop_back();
            continue; // on a constraint or hull edge: not insertable
          }
          _con_nbr.push_back({k_none, k_none});
          split_edge(host, on_edge, p, k_none, k_none);
        } else {
          _con_nbr.push_back({k_none, k_none});
          split_triangle(host, p);
        }
      }

      drain_one_generation();
    }
  }

  template <typename EdgesPolicy, typename IsSplittableRange>
  auto adopt(const tf::edges<EdgesPolicy> &edges,
             const IsSplittableRange &is_splittable) -> void {
    auto out = _cdt.converted_points();
    auto ints = _cdt.points();
    _ip.reserve(ints.size());
    _dp.reserve(ints.size());
    _con_nbr.reserve(ints.size());
    for (std::size_t i = 0; i < std::size_t(ints.size()); ++i) {
      _ip.push_back(tf::point<Int, 2>{ints[i][0], ints[i][1]});
      _dp.push_back({double(out[i][0]), double(out[i][1])});
      _con_nbr.push_back({k_none, k_none});
    }
    // affine double->int from the x-extreme pair (two arbitrary points risk
    // catastrophic cancellation when nearly x-coincident)
    std::size_t lo = 0, hi = 0;
    for (std::size_t i = 1; i < _dp.size(); ++i) {
      if (_dp[i][0] < _dp[lo][0])
        lo = i;
      if (_dp[i][0] > _dp[hi][0])
        hi = i;
    }
    double dx = _dp[hi][0] - _dp[lo][0];
    _scale = dx != 0 ? double(_ip[hi][0] - _ip[lo][0]) / dx : 1.0;
    _offx = double(_ip[lo][0]) - _dp[lo][0] * _scale;
    _offy = double(_ip[lo][1]) - _dp[lo][1] * _scale;

    // one sub-edge per input constraint, plus a sorted lookup by endpoints
    const auto &im = _cdt.index_map().f();
    auto &cons = _scratch_cons;
    cons.clear();
    cons.reserve(std::size_t(edges.size()));
    _segments.reserve(std::size_t(edges.size()));
    for (Index j = 0; j < _n_input_edges; ++j) {
      Index u = im[std::size_t(edges[j][0])];
      Index v = im[std::size_t(edges[j][1])];
      cons.push_back({std::min(u, v), std::max(u, v), j});
      _segments.push_back({j, u, 0,
                           is_splittable[std::size_t(j)]
                               ? std::uint8_t(0)
                               : std::uint8_t(k_max_split_depth)});
      auto hook = [&](Index x, Index y) {
        if (_con_nbr[std::size_t(x)][0] == k_none)
          _con_nbr[std::size_t(x)][0] = y;
        else
          _con_nbr[std::size_t(x)][1] = y;
      };
      hook(u, v);
      hook(v, u);
    }
    topology::detail::sort_auto(cons.begin(), cons.end());
    auto find_constraint = [&](Index u, Index v) -> Index {
      std::array<Index, 3> key{std::min(u, v), std::max(u, v), k_none};
      auto it = std::lower_bound(cons.begin(), cons.end(), key);
      if (it != cons.end() && (*it)[0] == key[0] && (*it)[1] == key[1])
        return (*it)[2];
      return k_none;
    };

    const auto n_faces = std::size_t(_cdt.n_triangles());
    _t.reserve(n_faces);
    _label.reserve(n_faces);
    // the CDT's DCEL already carries the adjacency: neighbors across opp,
    // constraint flags on edges. Its face handedness is uniform
    // (opp-consistent structure), so one orientation probe decides the
    // vertex order for every face.
    int handed = 0;
    _cdt.for_each_face_adjacency([&](Index i, Index a, Index b, Index c,
                                     Index label,
                                     const std::array<Index, 3> &nbrs,
                                     const std::array<bool, 3> &cons) {
      if (handed == 0)
        handed = orient(a, b, c) < 0 ? -1 : 1;
      triangle tt{};
      tt.stamp = 0;
      if (handed > 0) {
        tt.v[0] = a;
        tt.v[1] = b;
        tt.v[2] = c;
        for (int e = 0; e < 3; ++e) {
          tt.n[e] = nbrs[std::size_t(e)];
          tt.seg[e] = k_none;
        }
        for (int e = 0; e < 3; ++e)
          if (cons[std::size_t(e)])
            tt.seg[e] = find_constraint(tt.v[e], tt.v[(e + 1) % 3]);
      } else {
        // reversed: (a, c, b); edge k of the new order is the reverse of
        // DCEL edge (2 - k)
        tt.v[0] = a;
        tt.v[1] = c;
        tt.v[2] = b;
        for (int e = 0; e < 3; ++e) {
          tt.n[e] = nbrs[std::size_t(2 - e)];
          tt.seg[e] = k_none;
        }
        for (int e = 0; e < 3; ++e)
          if (cons[std::size_t(2 - e)])
            tt.seg[e] = find_constraint(tt.v[e], tt.v[(e + 1) % 3]);
      }
      (void)i;
      _t.push_back(tt);
      _label.push_back(label);
    });
  }

  auto finalize_splits() -> void {
    topology::detail::sort_auto(
        _splits.begin(), _splits.end(),
        [](const cdt_constraint_split<Index> &a,
           const cdt_constraint_split<Index> &b) {
          if (a.edge != b.edge)
            return a.edge < b.edge;
          return (std::uint32_t(a.numerator)
                  << (k_max_split_depth + 1 - a.depth)) <
                 (std::uint32_t(b.numerator)
                  << (k_max_split_depth + 1 - b.depth));
        });
    _split_offsets.allocate(std::size_t(_n_input_edges) + 1);
    std::size_t w = 0;
    for (Index j = 0; j <= _n_input_edges; ++j) {
      _split_offsets[std::size_t(j)] = Index(w);
      while (w < _splits.size() && j < _n_input_edges &&
             _splits[w].edge == j)
        ++w;
    }
  }

  constrained_delaunay_triangulator<Index, Coord, Int> _cdt;
  // per-build scratch: reused across builds, never freed
  tf::buffer<std::array<Index, 3>> _scratch_cons;
  tf::buffer<tf::point<Int, 2>> _ip;
  tf::buffer<std::array<double, 2>> _dp;
  tf::buffer<triangle> _t;
  tf::buffer<Index> _label;
  tf::buffer<sub_edge> _segments;
  tf::buffer<std::array<Index, 2>> _con_nbr;
  tf::buffer<std::array<Index, 2>> _queue;
  tf::buffer<Index> _cavity;
  tf::buffer<Index> _lawson;
  tf::buffer<Index> _pending;
  tf::buffer<Index> _generation;
  tf::buffer<cdt_constraint_split<Index>> _splits;
  tf::buffer<Index> _split_offsets;
  double _scale = 1;
  double _offx = 0;
  double _offy = 0;
  Index _n_input_points = 0;
  Index _n_input_edges = 0;
  bool _split_encroached = true;
  double _min_quality = 0.0;
  bool _ok = false;
};

} // namespace tf

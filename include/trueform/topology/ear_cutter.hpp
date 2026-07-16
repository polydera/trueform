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
#include "../core/coordinate_type.hpp"
#include "../core/polygon.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "../exact/meta.hpp"
#include "../exact_coordinate_converter.hpp"
#include "../exact/signed_area.hpp"
#include <algorithm>
#include <utility>

namespace tf {

/// Exact ear-clipping triangulation for 2D polygons.
///
/// Accepts integral or float points. Float inputs are automatically
/// converted via pt_converter for exact arithmetic.
/// All predicates use exact arithmetic via meta<Int> (T1 differences,
/// T2 products). Positive area = CCW.
///
/// build(ids, points): ids are identity indices into the points
/// buffer. Handles pseudosimple polygons from hole bridging (doubled
/// bridge vertices with opposite-direction edges).
///
/// Invariants:
///   1. Every input vertex appears in at least one output triangle.
///   2. Every boundary edge appears exactly once in the triangulation.
///   3. Every interior edge appears exactly twice (opposite directions).
///   4. Every emitted triangle has 3 distinct pt_ids (identities).
///      Exception: outward shards, where the degenerate ear is the
///      only way to include the shard tip vertex.
///
/// Zero-area triangles occur only for outward shards and residual
/// collinear runs (2-node edge with absorbed collinear points,
/// fan-emitted to satisfy invariant 1).
///
/// CW input is detected via signed_area_2x and reversed; output
/// triangles are flipped back to match the original winding.
/// Z-order hashing accelerates ear detection for polygons > 80 verts.
///
/// When earcut gets stuck (no ears after a full loop), fallbacks:
///   1. Resolve shards -- geometric slits where prev and next share
///      coordinates. Tries diagonal from the shard tip; if none
///      exists (outward shard), clips as degenerate ear.
///   2. Remove collinear runs -- collapses consecutive collinear
///      vertices into a linked list on their predecessor node.
///      Uses on_segment betweenness to stop at shard turnarounds.
///      Runs are fan-expanded on ear emission.
///   3. Resolve pinches -- a point the ring visits twice,
///      non-adjacently, with both would-be cycles of positive area
///      (a weakly-simple polygon: two lobes meeting at a point, e.g.
///      chained holes spliced at a shared vertex). The pair is not a
///      diagonal -- nothing is inserted; the ring is relinked into
///      two cycles, each keeping one occurrence. Separation is
///      exhausted across all resulting cycles before cutting resumes.
///      Keyhole bridges fail the positive-area gate (their split-off
///      cycle is the hole, wound negative) and stay whole.
///   4. Find a valid diagonal and split (last resort).
///
/// When an ear has runs on multiple edges, the last point of one run
/// is re-inserted and the polygon is split via diagonal, distributing
/// runs across two sub-polygons that are triangulated recursively.
template <typename Index, typename Int = tf::exact::int32> class ear_cutter {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

public:
  auto faces() const { return tf::make_blocked_range<3>(_triangles); }
  auto indices_buffer() const -> const tf::buffer<Index> & {
    return _triangles;
  }
  auto indices_buffer() -> tf::buffer<Index> & { return _triangles; }

  auto clear() -> void {
    _nodes.clear();
    _triangles.clear();
    _removed.clear();
    _pinch_cycles.clear();
    _in_pinch = false;
    _hashing = false;
    _min_x = _max_x = _min_y = _max_y = 0;
    _z_shift = 0;
  }

  template <typename Ids, typename Policy>
  auto build(const Ids &ids, const tf::points<Policy> &points) -> bool {
    using coord_t = tf::coordinate_type<std::decay_t<decltype(points)>>;
    if constexpr (std::is_integral_v<coord_t>) {
      return run(ids, points);
    } else {
      auto conv = tf::make_exact_coordinate_converter<Int>(points);
      auto int_pts = tf::make_points(tf::make_mapped_range(
          points, [&](const auto &pt) { return conv(pt); }));
      return run(ids, int_pts);
    }
  }

  template <typename Policy>
  auto build(const tf::points<Policy> &points) -> bool {
    return build(tf::make_sequence_range(points.size()), points);
  }

private:
  template <typename Ids, typename Points>
  auto run(const Ids &ids, const Points &points) -> bool {
    clear();
    auto n_pts = ids.size();
    if (n_pts <= 2) {
      if (n_pts <= 1)
        return false;
      _triangles.allocate(3);
      _triangles[0] = ids[0];
      _triangles[1] = ids[1];
      _triangles[2] = ids[0];
      return false;
    }
    if (n_pts == 3) {
      _triangles.allocate(3);
      _triangles[0] = ids[0];
      _triangles[1] = ids[1];
      _triangles[2] = ids[2];
      return true;
    }

    Index size = static_cast<Index>(n_pts);
    _nodes.reserve(size + 8);
    _nodes.allocate(size);
    _triangles.reserve(static_cast<std::size_t>((size - 2) * 3));

    for (Index i = 0; i < size; ++i) {
      _nodes[i].id = i;
      _nodes[i].pt_id = ids[i];
      _nodes[i].prev = (i == 0) ? size - 1 : i - 1;
      _nodes[i].next = (i == size - 1) ? 0 : i + 1;
      _nodes[i].run_first = Index(-1);
      _nodes[i].run_last = Index(-1);
      _nodes[i].z = 0;
      _nodes[i].prev_z = Index(-1);
      _nodes[i].next_z = Index(-1);
    }

    bool reversed =
        tf::exact::signed_area_2x(tf::make_polygon(ids, points)) < 0;
    if (reversed) {
      for (Index i = 0; i < size; ++i)
        std::swap(_nodes[i].prev, _nodes[i].next);
    }

    _hashing = n_pts > 80;
    if (_hashing)
      compute_aabb(points, 0);

    bool ok = triangulate(points, 0);

    if (reversed) {
      for (std::size_t i = 0; i < _triangles.size(); i += 3)
        std::swap(_triangles[i], _triangles[i + 2]);
    }
    return ok;
  }

  struct node_t {
    Index id, pt_id, prev, next;
    Index run_first, run_last; // -1 if no collinear run
    T1 z;                     // z-order Morton code
    Index prev_z, next_z;      // z-order linked list
  };

  struct removed_node_t {
    Index pt_id;
    Index next; // -1 = end of chain
  };

  auto next(const node_t &n) const -> const node_t & { return _nodes[n.next]; }
  auto prev(const node_t &n) const -> const node_t & { return _nodes[n.prev]; }
  auto next(const node_t &n) -> node_t & { return _nodes[n.next]; }
  auto prev(const node_t &n) -> node_t & { return _nodes[n.prev]; }

  auto has_run(const node_t &n) const -> bool {
    return n.run_first != Index(-1);
  }

  auto alloc_removed(Index pt_id) -> Index {
    Index idx = static_cast<Index>(_removed.size());
    _removed.push_back({pt_id, Index(-1)});
    return idx;
  }

  // ── Z-order hashing for spatial acceleration ──

  static auto z_order(T1 x, T1 y) -> T1 {
    x = (x | (x << 8)) & 0x00FF00FF;
    x = (x | (x << 4)) & 0x0F0F0F0F;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    y = (y | (y << 8)) & 0x00FF00FF;
    y = (y | (y << 4)) & 0x0F0F0F0F;
    y = (y | (y << 2)) & 0x33333333;
    y = (y | (y << 1)) & 0x55555555;
    return x | (y << 1);
  }

  template <typename Pts>
  auto compute_aabb(const Pts &pts, Index start) -> void {
    auto &&p0 = pts[_nodes[start].pt_id];
    _min_x = _max_x = p0[0];
    _min_y = _max_y = p0[1];
    Index c = _nodes[start].next;
    while (c != start) {
      auto &&p = pts[_nodes[c].pt_id];
      _min_x = std::min(_min_x, p[0]);
      _max_x = std::max(_max_x, p[0]);
      _min_y = std::min(_min_y, p[1]);
      _max_y = std::max(_max_y, p[1]);
      c = _nodes[c].next;
    }
    T1 range = std::max(T1(_max_x) - T1(_min_x), T1(_max_y) - T1(_min_y));
    _z_shift = 0;
    if (range > 0) {
      auto r = static_cast<typename tf::exact::meta<Int>::unsigned_T1>(range);
      while (r > 32767) {
        r >>= 1;
        ++_z_shift;
      }
    }
  }

  auto to_z(Int x, Int y) const -> T1 {
    T1 nx = (T1(x) - T1(_min_x)) >> _z_shift;
    T1 ny = (T1(y) - T1(_min_y)) >> _z_shift;
    return z_order(nx, ny);
  }

  template <typename Pts>
  auto index_curve(const Pts &pts, Index start) -> void {
    Index c = start;
    do {
      auto &n = _nodes[c];
      if (n.z == 0) {
        auto &&p = pts[n.pt_id];
        n.z = to_z(p[0], p[1]);
      }
      n.prev_z = n.prev;
      n.next_z = n.next;
      c = n.next;
    } while (c != start);
    // Break circular z-list into linear for sorting
    _nodes[_nodes[start].prev_z].next_z = Index(-1);
    _nodes[start].prev_z = Index(-1);
    sort_linked_z(start);
  }

  auto sort_linked_z(Index list_id) -> Index {
    Index p_id, q_id, e_id, tail_id;
    int i, num_merges, p_size, q_size;
    int in_size = 1;
    for (;;) {
      p_id = list_id;
      list_id = Index(-1);
      tail_id = Index(-1);
      num_merges = 0;
      while (p_id != Index(-1)) {
        num_merges++;
        q_id = p_id;
        p_size = 0;
        for (i = 0; i < in_size; ++i) {
          p_size++;
          q_id = _nodes[q_id].next_z;
          if (q_id == Index(-1))
            break;
        }
        q_size = in_size;
        while (p_size > 0 || (q_size > 0 && q_id != Index(-1))) {
          if (p_size == 0) {
            e_id = q_id;
            q_id = _nodes[q_id].next_z;
            q_size--;
          } else if (q_size == 0 || q_id == Index(-1)) {
            e_id = p_id;
            p_id = _nodes[p_id].next_z;
            p_size--;
          } else if (_nodes[p_id].z <= _nodes[q_id].z) {
            e_id = p_id;
            p_id = _nodes[p_id].next_z;
            p_size--;
          } else {
            e_id = q_id;
            q_id = _nodes[q_id].next_z;
            q_size--;
          }
          if (tail_id != Index(-1))
            _nodes[tail_id].next_z = e_id;
          else
            list_id = e_id;
          _nodes[e_id].prev_z = tail_id;
          tail_id = e_id;
        }
        p_id = q_id;
      }
      _nodes[tail_id].next_z = Index(-1);
      if (num_merges <= 1)
        return list_id;
      in_size *= 2;
    }
  }

  /// Absorb node v into the run on node a.
  auto absorb_into_run(node_t &a, const node_t &v) -> void {
    auto v_id = alloc_removed(v.pt_id);
    if (has_run(v)) {
      _removed[v_id].next = v.run_first;
      if (has_run(a)) {
        _removed[a.run_last].next = v_id;
      } else {
        a.run_first = v_id;
      }
      a.run_last = v.run_last;
    } else {
      if (has_run(a)) {
        _removed[a.run_last].next = v_id;
      } else {
        a.run_first = v_id;
      }
      a.run_last = v_id;
    }
  }

  /// Remove collinear node v, absorbing it into prev(v)'s run.
  /// Returns {prev_id, next_id} after unlinking.
  auto absorb_collinear(node_t &v) -> std::pair<Index, Index> {
    auto pv_id = v.prev;
    auto nc_id = v.next;
    absorb_into_run(_nodes[pv_id], v);
    _nodes[pv_id].next = nc_id;
    _nodes[nc_id].prev = pv_id;
    return {pv_id, nc_id};
  }

  // ── Exact predicates (Int → T1/T2) ──

  template <typename Pts>
  auto area(const Pts &pts, const node_t &p, const node_t &q,
            const node_t &r) const -> T2 {
    auto &&pp = pts[p.pt_id];
    auto &&pq = pts[q.pt_id];
    auto &&pr = pts[r.pt_id];
    return T2(T1(pq[0]) - T1(pp[0])) * T2(T1(pr[1]) - T1(pp[1])) -
           T2(T1(pq[1]) - T1(pp[1])) * T2(T1(pr[0]) - T1(pp[0]));
  }

  template <typename Pts>
  auto contains_point(const Pts &pts, const node_t &a, const node_t &b,
                      const node_t &c, const node_t &p) const -> bool {
    auto cross = [](T1 ax, T1 ay, T1 bx, T1 by, T1 px, T1 py) -> T2 {
      return T2(bx - px) * T2(ay - py) - T2(ax - px) * T2(by - py);
    };
    auto &&pa = pts[a.pt_id];
    auto &&pb = pts[b.pt_id];
    auto &&pc = pts[c.pt_id];
    auto &&pp = pts[p.pt_id];
    return cross(pc[0], pc[1], pa[0], pa[1], pp[0], pp[1]) <= 0 &&
           cross(pa[0], pa[1], pb[0], pb[1], pp[0], pp[1]) <= 0 &&
           cross(pb[0], pb[1], pc[0], pc[1], pp[0], pp[1]) <= 0;
  }

  template <typename Pts>
  auto on_segment(const Pts &pts, const node_t &p, const node_t &q,
                  const node_t &r) const -> bool {
    auto &&pp = pts[p.pt_id];
    auto &&pq = pts[q.pt_id];
    auto &&pr = pts[r.pt_id];
    return pq[0] <= std::max(pp[0], pr[0]) && pq[0] >= std::min(pp[0], pr[0]) &&
           pq[1] <= std::max(pp[1], pr[1]) && pq[1] >= std::min(pp[1], pr[1]);
  }

  template <typename Pts>
  auto intersects(const Pts &pts, const node_t &p1, const node_t &q1,
                  const node_t &p2, const node_t &q2) const -> bool {
    auto sign = [](T2 v) -> int { return (v > 0) - (v < 0); };
    int o1 = sign(area(pts, p1, q1, p2));
    int o2 = sign(area(pts, p1, q1, q2));
    int o3 = sign(area(pts, p2, q2, p1));
    int o4 = sign(area(pts, p2, q2, q1));
    if (o1 != o2 && o3 != o4)
      return true;
    if (o1 == 0 && on_segment(pts, p1, p2, q1))
      return true;
    if (o2 == 0 && on_segment(pts, p1, q2, q1))
      return true;
    if (o3 == 0 && on_segment(pts, p2, p1, q2))
      return true;
    if (o4 == 0 && on_segment(pts, p2, q1, q2))
      return true;
    return false;
  }

  template <typename Pts>
  auto intersects_polygon(const Pts &pts, const node_t &a,
                          const node_t &b) const -> bool {
    Index p_id = a.id;
    do {
      auto &p = _nodes[p_id];
      if (p.pt_id != a.pt_id && next(p).pt_id != a.pt_id &&
          p.pt_id != b.pt_id && next(p).pt_id != b.pt_id &&
          intersects(pts, p, next(p), a, b))
        return true;
      p_id = next(p).id;
    } while (p_id != a.id);
    return false;
  }

  template <typename Pts>
  auto locally_inside(const Pts &pts, const node_t &a, const node_t &b) const
      -> bool {
    return area(pts, prev(a), a, next(a)) > 0
               ? area(pts, a, b, next(a)) <= 0 && area(pts, a, prev(a), b) <= 0
               : area(pts, a, b, prev(a)) > 0 || area(pts, a, next(a), b) > 0;
  }

  template <typename Pts>
  auto middle_inside(const Pts &pts, const node_t &a, const node_t &b) const
      -> bool {
    Index p_id = a.id;
    bool inside = false;
    T1 mx = T1(pts[a.pt_id][0]) + T1(pts[b.pt_id][0]);
    T1 my = T1(pts[a.pt_id][1]) + T1(pts[b.pt_id][1]);
    do {
      auto &p = _nodes[p_id];
      T1 py = 2 * T1(pts[p.pt_id][1]);
      T1 pny = 2 * T1(pts[next(p).pt_id][1]);
      if ((py > my) != (pny > my) && pny != py) {
        T1 px = pts[p.pt_id][0];
        T1 pnx = pts[next(p).pt_id][0];
        T2 dy = T2(pny - py);
        T2 lhs = T2(mx) * dy - T2(2 * px) * dy;
        T2 rhs = T2(pnx - px) * T2(my - py);
        if (dy < 0) {
          lhs = -lhs;
          rhs = -rhs;
        }
        if (lhs < rhs)
          inside = !inside;
      }
      p_id = next(p).id;
    } while (p_id != a.id);
    return inside;
  }

  template <typename Pts>
  auto is_valid_diagonal(const Pts &pts, const node_t &a, const node_t &b) const
      -> bool {
    bool mid = middle_inside(pts, a, b);
    bool loc =
        locally_inside(pts, a, b) && locally_inside(pts, b, a) && mid &&
        (area(pts, prev(a), a, prev(b)) != 0 || area(pts, a, prev(b), b) != 0);
    bool dup = (a.pt_id == b.pt_id && area(pts, prev(a), a, next(a)) < 0 &&
                area(pts, prev(b), b, next(b)) < 0);
    bool no_cross = next(a).pt_id != b.pt_id && prev(a).pt_id != b.pt_id &&
                    !intersects_polygon(pts, a, b);
    return no_cross && (loc || dup);
  }

  // ── Ear test ──

  template <typename Pts>
  auto is_ear(const node_t &v, const Pts &pts) const -> bool {
    if (area(pts, prev(v), v, next(v)) <= 0)
      return false;
    auto cid = next(next(v)).id;
    while (cid != prev(v).id) {
      if (contains_point(pts, prev(v), v, next(v), _nodes[cid]) &&
          area(pts, prev(_nodes[cid]), _nodes[cid], next(_nodes[cid])) <= 0)
        return false;
      cid = next(_nodes[cid]).id;
    }
    return true;
  }

  template <typename Pts>
  auto is_ear_hashed(const node_t &v, const Pts &pts) const -> bool {
    if (area(pts, prev(v), v, next(v)) <= 0)
      return false;
    auto &pv = prev(v);
    auto &nv = next(v);
    auto &&pa = pts[pv.pt_id];
    auto &&pb = pts[v.pt_id];
    auto &&pc = pts[nv.pt_id];
    auto min_tx = std::min({pa[0], pb[0], pc[0]});
    auto min_ty = std::min({pa[1], pb[1], pc[1]});
    auto max_tx = std::max({pa[0], pb[0], pc[0]});
    auto max_ty = std::max({pa[1], pb[1], pc[1]});
    T1 min_z = to_z(min_tx, min_ty);
    T1 max_z = to_z(max_tx, max_ty);

    // Search forward in z-order
    Index p = v.next_z;
    while (p != Index(-1) && _nodes[p].z <= max_z) {
      if (_nodes[p].id != pv.id && _nodes[p].id != nv.id &&
          contains_point(pts, pv, v, nv, _nodes[p]) &&
          area(pts, prev(_nodes[p]), _nodes[p], next(_nodes[p])) <= 0)
        return false;
      p = _nodes[p].next_z;
    }
    // Search backward in z-order
    p = v.prev_z;
    while (p != Index(-1) && _nodes[p].z >= min_z) {
      if (_nodes[p].id != pv.id && _nodes[p].id != nv.id &&
          contains_point(pts, pv, v, nv, _nodes[p]) &&
          area(pts, prev(_nodes[p]), _nodes[p], next(_nodes[p])) <= 0)
        return false;
      p = _nodes[p].prev_z;
    }
    return true;
  }

  // ── Emit ear ──

  auto emit_triangle(Index a, Index b, Index c) -> void {
    _triangles.push_back(a);
    _triangles.push_back(b);
    _triangles.push_back(c);
  }

  /// Fan-expand a collinear run along edge (from → to), walking the
  /// removed linked list from n.run_first. Apex is the opposite vertex.
  auto emit_fan(Index apex, Index from, const node_t &n) -> void {
    Index cur = n.run_first;
    Index prev_pt = from;
    while (cur != Index(-1)) {
      emit_triangle(apex, prev_pt, _removed[cur].pt_id);
      prev_pt = _removed[cur].pt_id;
      cur = _removed[cur].next;
    }
  }

  /// Process a collinear run: fan-triangulate the interior from apex,
  /// return (first_pt, last_pt) for the two sub-triangle nodes.
  auto process_run(Index apex, const node_t &owner)
      -> std::pair<Index, Index> {
    Index first_pt = _removed[owner.run_first].pt_id;
    Index last_pt = _removed[owner.run_last].pt_id;
    Index cur = owner.run_first;
    while (_removed[cur].next != Index(-1)) {
      Index nxt = _removed[cur].next;
      emit_triangle(apex, _removed[cur].pt_id, _removed[nxt].pt_id);
      cur = nxt;
    }
    return {first_pt, last_pt};
  }

  /// Recursively triangulate a 3-node loop, handling collinear runs.
  /// The triangle is (a, v, b) where a=prev(v), b=next(v).
  auto resolve_triangle(Index v_id) -> void {
    auto a_id = _nodes[v_id].prev;
    auto b_id = _nodes[v_id].next;

    auto split_and_recurse = [&](Index owner_id, Index first_pt,
                                 Index last_pt) {
      Index next_id = _nodes[owner_id].next;
      Index opp_id = _nodes[owner_id].prev;

      auto saved_owner = _nodes[owner_id];
      auto saved_next = _nodes[next_id];
      auto saved_opp = _nodes[opp_id];

      Index n_id = static_cast<Index>(_nodes.size());
      _nodes.push_back({n_id, first_pt, Index(-1), Index(-1),
                         Index(-1), Index(-1), 0, Index(-1), Index(-1)});

      // Sub-triangle 1: (owner, n_node=first, opp)
      _nodes[owner_id].next = n_id;
      _nodes[owner_id].prev = opp_id;
      _nodes[n_id].prev = owner_id;
      _nodes[n_id].next = opp_id;
      _nodes[opp_id].prev = n_id;
      _nodes[opp_id].next = owner_id;
      resolve_triangle(n_id);

      // Restore state before second split
      _nodes[owner_id] = saved_owner;
      _nodes[next_id] = saved_next;
      _nodes[opp_id] = saved_opp;

      // Reuse n_node for last_pt in sub-triangle 2: (n_node=last, next, opp)
      _nodes[n_id].pt_id = last_pt;
      _nodes[n_id].run_first = Index(-1);
      _nodes[n_id].run_last = Index(-1);
      _nodes[n_id].next = next_id;
      _nodes[n_id].prev = opp_id;
      _nodes[next_id].prev = n_id;
      _nodes[next_id].next = opp_id;
      _nodes[opp_id].prev = next_id;
      _nodes[opp_id].next = n_id;
      resolve_triangle(next_id);

      // Final restore
      _nodes[owner_id] = saved_owner;
      _nodes[next_id] = saved_next;
      _nodes[opp_id] = saved_opp;
    };

    if (has_run(_nodes[a_id])) {
      auto [first_pt, last_pt] = process_run(_nodes[b_id].pt_id, _nodes[a_id]);
      _nodes[a_id].run_first = _nodes[a_id].run_last = Index(-1);
      split_and_recurse(a_id, first_pt, last_pt);
    } else if (has_run(_nodes[v_id])) {
      auto [first_pt, last_pt] = process_run(_nodes[a_id].pt_id, _nodes[v_id]);
      _nodes[v_id].run_first = _nodes[v_id].run_last = Index(-1);
      split_and_recurse(v_id, first_pt, last_pt);
    } else if (has_run(_nodes[b_id])) {
      auto [first_pt, last_pt] = process_run(_nodes[v_id].pt_id, _nodes[b_id]);
      _nodes[b_id].run_first = _nodes[b_id].run_last = Index(-1);
      split_and_recurse(b_id, first_pt, last_pt);
    } else {
      emit_triangle(_nodes[a_id].pt_id, _nodes[v_id].pt_id,
                     _nodes[b_id].pt_id);
    }
  }

  /// Clip ear v from the polygon and emit its triangles via
  /// resolve_triangle. Saves/restores a and b around the resolve so
  /// the polygon linked list stays intact.
  auto emit_ear(node_t &v) -> void {
    Index a_id = v.prev;
    Index b_id = v.next;

    if (_nodes[b_id].next == a_id) {
      // Already a 3-node loop — resolve directly, then clear runs
      auto saved_a = _nodes[a_id];
      auto saved_b = _nodes[b_id];
      resolve_triangle(v.id);
      _nodes[a_id] = saved_a;
      _nodes[b_id] = saved_b;
      _nodes[a_id].run_first = _nodes[a_id].run_last = Index(-1);
      _nodes[b_id].run_first = _nodes[b_id].run_last = Index(-1);
      _nodes[a_id].next = b_id;
      _nodes[b_id].prev = a_id;
      return;
    }

    // Polygon has more than 3 nodes
    auto saved_a = _nodes[a_id];
    auto saved_b = _nodes[b_id];
    _nodes[b_id].run_first = _nodes[b_id].run_last = Index(-1);
    _nodes[a_id].prev = b_id;
    _nodes[b_id].next = a_id;
    resolve_triangle(v.id);
    _nodes[a_id] = saved_a;
    _nodes[b_id] = saved_b;
    _nodes[a_id].run_first = _nodes[a_id].run_last = Index(-1);
    _nodes[a_id].next = b_id;
    _nodes[b_id].prev = a_id;
  }

 auto split_polygon(Index a_id, Index b_id) -> Index {
    Index n_id = static_cast<Index>(_nodes.size());
    auto na = _nodes[a_id];
    auto nb = _nodes[b_id];
    _nodes.push_back(na);
    _nodes.push_back(nb);
    auto &a = _nodes[a_id];
    auto &b = _nodes[b_id];

    // "left" loop gets a and b
    a.prev = b.id;
    //
    b.next = a.id;
    // since next becomes a, no runs on this edge
    b.run_first = Index(-1);
    b.run_last = Index(-1);

    // "right" loop gets a2 and b2
    Index a2_id = n_id;
    Index b2_id = n_id + 1;
    auto &a2 = _nodes[a2_id];
    // change only the modified values
    a2.id = a2_id;
    a2.next = b2_id;
    // since next becomes b2, no runs on this edge
    a2.run_first = Index(-1);
    a2.run_last = Index(-1);
    //
    prev(na).next = a2_id;
    //
    auto &b2 = _nodes[b2_id];
    // change only the modified values
    b2.id = b2_id;
    b2.prev = a2_id;
    // next is the old next. so we carry over runs
    b2.run_first = nb.run_first;
    b2.run_last = nb.run_last;
    //
    next(nb).prev = b2_id;
    return n_id + 1;
  } 

  // ── Phase 1: Shard resolution ──

  template <typename Pts>
  auto find_diagonal_from(const Pts &pts, Index b_id) -> Index {
    Index v = _nodes[b_id].next;
    auto start = v;
    do {
      if (v != b_id && _nodes[v].pt_id != _nodes[b_id].pt_id &&
          _nodes[v].next != b_id && _nodes[b_id].next != v &&
          is_valid_diagonal(pts, _nodes[b_id], _nodes[v]))
        return v;
      v = _nodes[v].next;
    } while (v != start);
    return Index(-1);
  }

  /// Resolve geometric slits (prev and next share coordinates).
  /// Tries to find a diagonal from the shard tip and split. If no
  /// diagonal exists (outward shard), clips as degenerate ear.
  template <typename Pts>
  auto resolve_shards(const Pts &pts, Index start) -> Index {
    Index c = start;
    do {
      auto &cv = _nodes[c];
      if (cv.prev == cv.next)
        break;
      if (pts[_nodes[cv.prev].pt_id] != pts[_nodes[cv.next].pt_id]) {
        c = cv.next;
        continue;
      }
      if (pts[cv.pt_id] == pts[_nodes[cv.prev].pt_id]) {
        c = cv.next;
        continue;
      }
      Index v = find_diagonal_from(pts, c);
      if (v != Index(-1)) {
        Index other = split_polygon(c, v);
        triangulate(pts, c);
        triangulate(pts, other);
        return Index(-1);
      }
      emit_ear(cv);
      auto nc = cv.next;
      if (c == start)
        start = nc;
      c = nc;
    } while (c != start);
    return start;
  }

  // ── Pinch separation (last-chance fallback on a stuck ring) ──

  /// Signed area (2x) of the cycle that would run from `from` up to,
  /// but not including, `to` if the ring were separated there.
  template <typename Pts>
  auto cycle_area(const Pts &pts, Index from, Index to) const -> T2 {
    T2 acc = 0;
    for (Index c = _nodes[from].next; _nodes[c].next != to;
         c = _nodes[c].next)
      acc += area(pts, _nodes[from], _nodes[c], next(_nodes[c]));
    return acc;
  }

  /// Relink the ring into two cycles at a doubly-visited point.
  /// No nodes or edges are added; each cycle keeps one occurrence.
  /// Runs stay valid: only the prev links of a and b change, and the
  /// replacement predecessors hold the same coordinates.
  auto separate(Index a_id, Index b_id) -> void {
    Index pa = _nodes[a_id].prev;
    Index pb = _nodes[b_id].prev;
    _nodes[pb].next = a_id;
    _nodes[a_id].prev = pb;
    _nodes[pa].next = b_id;
    _nodes[b_id].prev = pa;
  }

  /// On a stuck ring, separate every coincident non-adjacent pair
  /// whose two would-be cycles both keep positive area (a pinch: two
  /// lobes meeting at a doubly-visited point, e.g. chained holes
  /// spliced at a shared vertex). Separation is exhausted across all
  /// resulting cycles BEFORE any cutting resumes — resuming early
  /// lets ears eat through a remaining pinch and strand degenerate
  /// residue. The pair is not a diagonal: nothing is inserted.
  /// Keyhole bridges fail the sign gate (their split-off cycle is the
  /// hole, wound negative) and stay whole. Returns true if a pinch
  /// was separated; the pieces' results fold into `completed`.
  template <typename Pts>
  auto resolve_pinches(const Pts &pts, Index start, bool &completed)
      -> bool {
    // Pieces cut after exhaustion are pinch-free by construction:
    // shards and collinear runs create no coincidences, and split
    // copies are adjacent, which the scan excludes. A nested stall
    // must not rebuild the worklist mid-iteration.
    if (_in_pinch)
      return false;
    _pinch_cycles.clear();
    _pinch_cycles.push_back(start);
    bool any = false;
    for (std::size_t w = 0; w < _pinch_cycles.size(); ++w) {
      for (bool again = true; again;) {
        again = false;
        const Index start_w = _pinch_cycles[w];
        Index a_id = start_w;
        do {
          for (Index b_id = _nodes[_nodes[a_id].next].next;
               b_id != _nodes[a_id].prev && !again;
               b_id = _nodes[b_id].next) {
            if (pts[_nodes[b_id].pt_id] != pts[_nodes[a_id].pt_id])
              continue;
            if (cycle_area(pts, a_id, b_id) > 0 &&
                cycle_area(pts, b_id, a_id) > 0) {
              separate(a_id, b_id);
              _pinch_cycles[w] = a_id;
              _pinch_cycles.push_back(b_id);
              any = again = true;
            }
          }
          a_id = _nodes[a_id].next;
        } while (a_id != start_w && !again);
      }
    }
    if (!any)
      return false;
    _in_pinch = true;
    for (std::size_t k = 0; k < _pinch_cycles.size(); ++k)
      completed &= earcut(pts, _pinch_cycles[k]);
    _in_pinch = false;
    return true;
  }

  // ── Collinear run removal (called when earcut gets stuck) ──

  template <typename Pts>
  auto remove_collinear_runs(const Pts &pts, Index start) -> Index {
    bool removed_any = false;
    bool again = true;
    while (again) {
      again = false;
      Index c = start;
      do {
        auto &v = _nodes[c];
        Index nc = v.next;
        if (v.prev != v.next && area(pts, prev(v), v, next(v)) == 0 &&
            on_segment(pts, prev(v), v, next(v))) {
          absorb_collinear(v);
          if (c == start)
            start = nc;
          removed_any = true;
          again = true;
        }
        c = nc;
      } while (c != start);
    }
    return removed_any ? start : Index(-1);
  }

  // ── Split earcut fallback ──

  template <typename Pts>
  auto split_earcut(const Pts &pts, Index start_id) -> bool {
    auto a = start_id;
    do {
      auto b = next(next(_nodes[a])).id;
      while (b != prev(_nodes[a]).id) {
        if (_nodes[a].pt_id != _nodes[b].pt_id &&
            is_valid_diagonal(pts, _nodes[a], _nodes[b])) {
          auto c_id = split_polygon(a, b);
          bool success = triangulate(pts, a);
          success &= triangulate(pts, c_id);
          return success;
        }
        b = next(_nodes[b]).id;
      }
      a = next(_nodes[a]).id;
    } while (a != start_id);
    return false;
  }

  // ── Earcut main loop ──

  template <typename Pts>
  auto earcut(const Pts &pts, Index start_id, int pass = 0) -> bool {
    if (!pass && _hashing)
      index_curve(pts, start_id);
    auto end_id = start_id;
    auto current_id = start_id;
    bool completed = true;

    while (_nodes[current_id].prev != _nodes[current_id].next) {
      auto &cur = _nodes[current_id];
      bool ear = _hashing ? is_ear_hashed(cur, pts) : is_ear(cur, pts);
      if (ear) {
        emit_ear(_nodes[current_id]);
        current_id = next(next(_nodes[current_id])).id;
        end_id = current_id;
        continue;
      }
      if (next(next(next(_nodes[current_id]))).id == current_id) {
        emit_ear(_nodes[current_id]);
        return false;
      }
      current_id = next(_nodes[current_id]).id;
      if (current_id == end_id) {
        // 1. Shards
        Index shard_result = resolve_shards(pts, current_id);
        if (shard_result == Index(-1)) {
          completed = true;
          break;
        }
        // 2. Collinear runs
        if (!pass) {
          Index new_start = remove_collinear_runs(pts, shard_result);
          if (new_start != Index(-1)) {
            completed &= earcut(pts, new_start, 1);
            break;
          }
        }
        // 3. Pinches (doubly-visited points → separate the lobes)
        if (resolve_pinches(pts, shard_result, completed))
          break;
        // 4. Diagonal split as last resort
        completed &= split_earcut(pts, shard_result);
        break;
      }
    }
    // Emit residual collinear runs on the remaining 2-node edge.
    // This handles fully-collinear polygons where earcut leaves
    // 2 nodes with absorbed collinear points in their runs.
    emit_residual_runs(current_id);
    return completed;
  }

  auto emit_residual_runs(Index start) -> void {
    auto &a = _nodes[start];
    if (a.prev != a.next)
      return; // more than 2 nodes, nothing to do
    auto &b = _nodes[a.next];
    if (has_run(a)) {
      emit_fan(b.pt_id, a.pt_id, a);
      a.run_first = a.run_last = Index(-1);
    }
    if (has_run(b)) {
      emit_fan(a.pt_id, b.pt_id, b);
      b.run_first = b.run_last = Index(-1);
    }
  }

  /// Entry point: resolve shards first, then earcut.
  template <typename Pts>
  auto triangulate(const Pts &pts, Index start) -> bool {
    if (_nodes[start].next == start || _nodes[_nodes[start].next].next == start)
      return true;

    Index clean = resolve_shards(pts, start);
    if (clean == Index(-1))
      return true;

    if (_nodes[clean].next == clean || _nodes[_nodes[clean].next].next == clean)
      return true;

    return earcut(pts, clean);
  }

  // ── Data ──

  tf::buffer<node_t> _nodes;
  tf::buffer<Index> _triangles;
  tf::buffer<removed_node_t> _removed;
  tf::buffer<Index> _pinch_cycles; // stall fallback: cycle worklist
  Int _min_x = 0, _max_x = 0, _min_y = 0, _max_y = 0;
  int _z_shift = 0;
  bool _in_pinch = false;
  bool _hashing = false;
};

} // namespace tf

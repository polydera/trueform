/*
 * Copyright (c) 2026 XLAB
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
#include "../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/buffer.hpp"
#include "../core/edges.hpp"
#include "../core/frame_of.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/points_buffer.hpp"
#include "../core/range.hpp"
#include "../core/small_vector.hpp"
#include "../core/transformed.hpp"
#include "../core/views/constant.hpp"
#include "../core/views/sequence_range.hpp"
#include "../core/views/zip.hpp"
#include "../cut/loop_connectivity.hpp"
#include "../exact/projection_axes.hpp"
#include "../intersect/graph/vertex.hpp"
#include "../topology/cdt_refiner.hpp"
#include "../topology/constrained_delaunay_triangulator.hpp"
#include "./refine_sizing.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

namespace tf::cut {

namespace detail {

/// Physical-vertex identity, the house convention: `{-1, created id}`
/// for created points (tag-independent), `{tag, original id}` otherwise.
template <typename Index, typename V>
inline auto pkey(Index tag, const V &v) -> std::array<Index, 2> {
  return v.source == tf::intersect::graph::vertex_source::created
             ? std::array<Index, 2>{Index(-1), v.id}
             : std::array<Index, 2>{tag, v.id};
}
template <typename Index>
inline auto pkey_orig(Index tag, Index id) -> std::array<Index, 2> {
  return {tag, id};
}

/// One dyadic split on a physical edge; param measured from the min-key
/// endpoint in the 256-unit space.
template <typename Index> struct split_rec {
  std::array<Index, 2> ka, kb;
  std::uint8_t num, depth;
  auto param() const -> std::uint32_t {
    return std::uint32_t(num) << (8 - depth);
  }
  auto operator<(const split_rec &o) const -> bool {
    if (ka != o.ka)
      return ka < o.ka;
    if (kb != o.kb)
      return kb < o.kb;
    return param() < o.param();
  }
  auto operator==(const split_rec &o) const -> bool {
    return ka == o.ka && kb == o.kb && param() == o.param();
  }
};

// plan-encoded triangle corner refs: r >= 0 -> cycle corner index;
// r < 0, e = -r-1: e even -> split e/2, e odd -> steiner (e-1)/2
template <typename Index> constexpr auto split_ref(Index s) -> Index {
  return -(2 * s) - 1;
}
template <typename Index> constexpr auto steiner_ref(Index t) -> Index {
  return -(2 * t) - 2;
}

template <typename Index, typename Int> struct emit_scratch {
  tf::buffer<Index> og;
  tf::buffer<tf::point<Int, 3>> o3;
  tf::buffer<char> known;
};

/// CDT of (boundary cycle prefix n_cycle of lp3/lref + extra input points),
/// emit plan-encoded tris oriented to nd. New refiner insertions (qual > 0,
/// frozen) go to `new_steiner` with LOCAL ids; caller rebases at aggregation.
template <typename Index, typename Int>
inline auto cdt_emit(tf::cdt_refiner<Index, Int, Int> &rf,
                     tf::points_buffer<Int, 2> &rpts, tf::buffer<Index> &flat,
                     const tf::buffer<tf::point<Int, 3>> &lp3,
                     const tf::buffer<Index> &lref, std::size_t n_cycle,
                     int af, int as, const double nd[3], const double ad[3],
                     float qual, tf::buffer<std::array<Index, 3>> &out_tris,
                     tf::buffer<tf::point<Int, 3>> *new_steiner,
                     std::size_t &fans, emit_scratch<Index, Int> &sc)
    -> void {
  rpts.allocate(lp3.size());
  for (std::size_t i = 0; i < lp3.size(); ++i)
    rpts[i] = tf::point<Int, 2>{lp3[i][af], lp3[i][as]};
  flat.clear();
  for (std::size_t i = 0; i < n_cycle; ++i) {
    flat.push_back(Index(i));
    flat.push_back(Index((i + 1) % n_cycle));
  }
  tf::cdt_refine_config rc;
  rc.min_quality = qual;
  rc.split_encroached = false;
  bool ok = rf.build(rpts.points(), tf::make_edges(tf::make_range(flat)), rc);
  auto orient_emit = [&](Index a, Index b, Index c,
                         const tf::point<Int, 3> &pa,
                         const tf::point<Int, 3> &pb,
                         const tf::point<Int, 3> &pc) {
    double u0 = double(pb[0]) - pa[0], u1 = double(pb[1]) - pa[1],
           u2 = double(pb[2]) - pa[2];
    double v0 = double(pc[0]) - pa[0], v1 = double(pc[1]) - pa[1],
           v2 = double(pc[2]) - pa[2];
    double nx = u1 * v2 - u2 * v1, ny = u2 * v0 - u0 * v2,
           nz = u0 * v1 - u1 * v0;
    if (nx * nd[0] + ny * nd[1] + nz * nd[2] < 0)
      out_tris.push_back({a, c, b});
    else
      out_tris.push_back({a, b, c});
  };
  if (!ok) {
    ++fans; // watertight by ids; geometry may fold
    for (std::size_t i = 1; i + 1 < n_cycle; ++i)
      orient_emit(lref[0], lref[i], lref[i + 1], lp3[0], lp3[i], lp3[i + 1]);
    return;
  }
  auto rlabels = rf.region_labels();
  auto rp = rf.points();
  const auto &im = rf.index_map().f();
  auto &og = sc.og;
  auto &o3 = sc.o3;
  auto &known = sc.known;
  og.allocate(rp.size());
  o3.allocate(rp.size());
  known.allocate(rp.size());
  std::fill(known.begin(), known.end(), char(0));
  for (std::size_t i = 0; i < lp3.size(); ++i) {
    og[std::size_t(im[i])] = lref[i];
    o3[std::size_t(im[i])] = lp3[i];
    known[std::size_t(im[i])] = 1;
  }
  for (std::size_t i = 0; i < std::size_t(rf.n_faces()); ++i) {
    if (rlabels[i] % 2 != 1)
      continue;
    auto f = rf.face(Index(i));
    bool drop_face = false;
    for (int e = 0; e < 3; ++e) {
      std::size_t vi = std::size_t(f[e]);
      if (!known[vi]) {
        if (!new_steiner) {
          drop_face = true;
          continue;
        }
        Index sid = Index(new_steiner->size());
        auto p3 = lift3(rp[vi][0], rp[vi][1], af, as, nd, ad);
        new_steiner->push_back(p3);
        o3[vi] = p3;
        og[vi] = steiner_ref(sid); // LOCAL id
        known[vi] = 1;
      }
    }
    if (drop_face)
      continue;
    orient_emit(og[std::size_t(f[0])], og[std::size_t(f[1])],
                og[std::size_t(f[2])], o3[std::size_t(f[0])],
                o3[std::size_t(f[1])], o3[std::size_t(f[2])]);
  }
}

} // namespace detail

/// @ingroup csg
/// @brief Pre-computed per-loop triangulations, carried by @ref tf::csg_graph
///        and read by the csg extractions (triangulate once, extract many).
///
/// Triangle corners are @ref tf::intersect::graph::vertex — the same type
/// the face-cut loops carry — so extraction just `map_vertex()`es them.
/// Split and steiner points introduced by refinement are CREATED points:
/// their ids continue past the intersection graph's (`n_ig_created` is the
/// watermark), and their coordinates live in the graph's unified
/// created-points buffer, so downstream sees one uniform created id space.
/// Folded coplanar-stack loops ALIAS their representative's triangle range
/// (the redirect lives in `ranges`; `rev` flips winding), so one
/// triangulation serves the whole stack. Conforming uncut faces — uncut
/// faces that consume boundary splits from a cut neighbour — get their own
/// keyed triangle stream.
template <typename Index, typename Int> struct loop_triangulations {
  using vertex_t = tf::intersect::graph::vertex<Index>;

  tf::buffer<std::array<Index, 2>> ranges; // per loop [begin,end), aliasable
  tf::buffer<std::array<vertex_t, 3>> tris;
  tf::buffer<char> rev; // winding flip for folded loops
  // created ids below the watermark belong to the intersection graph;
  // at or above they index the provenance tables at id - n_ig_created
  Index n_ig_created = 0;
  // provenance per extra created point: kind 0 = split (parent = the
  // physical edge, a pair of identities), kind 1 = steiner (parent =
  // {{tag, face}, {-1, -1}})
  tf::buffer<char> extra_kind;
  tf::buffer<std::array<std::array<Index, 2>, 2>> extra_parent;
  // conforming uncut faces
  tf::buffer<std::array<Index, 2>> conf_keys; // sorted (tag, face)
  tf::buffer<Index> conf_tri_offsets;
  tf::buffer<std::array<vertex_t, 3>> conf_tris;
  // loops whose triangulation reported an incomplete cover
  Index n_failed = 0;

  auto loop_tris(Index li) const {
    auto r = ranges[std::size_t(li)];
    return tf::make_range(tris.begin() + r[0], tris.begin() + r[1]);
  }
  auto conf_index(Index tag, Index face) const -> Index {
    std::array<Index, 2> k{tag, face};
    auto it = std::lower_bound(conf_keys.begin(), conf_keys.end(), k);
    if (it == conf_keys.end() || !((*it)[0] == tag && (*it)[1] == face))
      return Index(-1);
    return Index(it - conf_keys.begin());
  }

  /// @brief Provenance of an extra created point.
  ///
  /// Created ids below `n_ig_created` are the intersection graph's own
  /// points (its provenance applies) — do not pass them here. At or above,
  /// the id indexes the provenance tables at `created_id - n_ig_created`.
  /// Returns `{kind, parent}`: kind 0 = split, parent = the two
  /// endpoint identities of the parent physical edge; kind 1 = steiner,
  /// parent = `{{tag, face}, {-1, -1}}`.
  auto created_origin(Index created_id) const
      -> std::pair<char, std::array<std::array<Index, 2>, 2>> {
    auto k = std::size_t(created_id - n_ig_created);
    return {extra_kind[k], extra_parent[k]};
  }

  /// Coplanar-stack canonicalization, mirroring the arrangement graph:
  /// every folded (dead) loop's range becomes an alias of its surviving
  /// representative's range; `rev` marks reversed winding.
  template <typename AG> auto apply_stack_aliases(const AG &ag) -> void {
    tf::parallel_fill(rev, char(0));
    // Cliques include (dead, dead) triples; only the direct
    // (live survivor, dead) pair carries the winding relative to the
    // triangles actually aliased.
    for (const auto &cp : ag.coplanar_pairs()) { // (survivor, dead, reversed)
      if (ag.dead_loops()[std::size_t(cp[0])])
        continue;
      ranges[std::size_t(cp[1])] = ranges[std::size_t(cp[0])];
      rev[std::size_t(cp[1])] = char(cp[2]);
    }
  }

  /// @brief Stock mode: plain Delaunay per live loop. No extra points,
  ///        empty conforming stream.
  template <typename AG, typename FC, typename IG, typename Forms,
            typename Converter>
  auto build_stock(const AG &ag, const FC &fc, const IG &ig,
                   const Forms &forms, const Converter &conv) -> void {
    auto descs = fc.descriptors();
    auto loops = fc.loops();
        const std::size_t n_loops = std::size_t(loops.size());
    n_ig_created = Index(ig.points().size());
    ranges.allocate(n_loops);
    rev.allocate(n_loops);
    auto ipts = ig.points();

    struct local_t {
      tf::constrained_delaunay_triangulator<Index, Int, Int> tri;
      tf::small_vector<tf::point<Int, 2>, 10> pts;
      tf::small_vector<Index, 24> cons;
      tf::buffer<Index> rep;
      tf::buffer<std::array<vertex_t, 3>> out;
      tf::buffer<Index> counts;
      Index fails = 0;
    };
    auto task = [&](auto &&range, local_t &local) {
      for (auto &&[li_z, desc, loop] : range) {
        std::size_t before = local.out.size();
        if (desc.tag != Index(-1) && loop.size() >= 3 &&
            !ag.dead_loops()[std::size_t(li_z)]) {
          const Index tag = desc.tag;
          auto face = forms[std::size_t(tag)].faces()[desc.object];
          auto get_pt = [&, tag](Index vid) -> tf::point<Int, 3> {
            return conv.convert(
                tf::transformed(forms[std::size_t(tag)].points()[vid],
                                tf::frame_of(forms[std::size_t(tag)])));
          };
          auto axes = tf::exact::projection_axes(
              get_pt(Index(face[0])), get_pt(Index(face[1])),
              get_pt(Index(face[2])));
          local.pts.clear();
          for (const auto &v : loop) {
            tf::point<Int, 3> pt =
                v.source == tf::intersect::graph::vertex_source::created
                    ? ipts[v.id]
                    : get_pt(v.id);
            local.pts.push_back({pt[axes.first], pt[axes.second]});
          }
          // Loop chords as region-boundary constraints, no ear cutting:
          // exact-equal projections weld (doubled bridge vertices), a
          // doubled edge toggles to a non-separating imprint, and the
          // interior is the parity-odd region. Winding follows the
          // loop's projected orientation.
          using T1 = typename tf::exact::meta<Int>::T1;
          using T2 = typename tf::exact::meta<Int>::T2;
          local.cons.clear();
          T2 area2(0);
          {
            const Index m = Index(local.pts.size());
            for (Index i = 0, j = m - 1; i < m; j = i++) {
              const auto &p = local.pts[std::size_t(j)];
              const auto &q = local.pts[std::size_t(i)];
              area2 += T2(T1(p[0]) * T1(q[1]) - T1(q[0]) * T1(p[1]));
              if (p[0] == q[0] && p[1] == q[1])
                continue;
              local.cons.push_back(j);
              local.cons.push_back(i);
            }
          }
          auto edges = tf::make_edges(tf::make_blocked_range<2>(
              tf::make_range(local.cons.data(), local.cons.size())));
          local.tri.clear();
          bool tri_ok = local.tri.build(
              tf::make_points(local.pts), edges,
              tf::make_constant_range(true, edges.size()),
              /*split_constraints=*/false);
          if (tri_ok) {
            // Welds are rare: with none, f() is a bijection and the
            // identity is unambiguous. Otherwise pick the canonical
            // representative per welded vertex — the smallest identity
            // pair — so callers sharing the identity space (stack
            // members, reversed windings) pick the same survivor
            // regardless of input order.
            const auto &fmap = local.tri.index_map().f();
            const Index n_final = Index(local.tri.points().size());
            const bool welded = std::size_t(n_final) != loop.size();
            local.rep.clear();
            if (welded) {
              local.rep.allocate(std::size_t(n_final));
              std::fill(local.rep.begin(), local.rep.end(), Index(-1));
              for (Index i = 0; i < Index(loop.size()); ++i) {
                const Index fin = fmap[std::size_t(i)];
                if (fin < 0 || n_final <= fin) {
                  tri_ok = false;
                  break;
                }
                auto &r = local.rep[std::size_t(fin)];
                if (r == Index(-1) ||
                    detail::pkey(tag, loop[std::size_t(i)]) <
                        detail::pkey(tag, loop[std::size_t(r)]))
                  r = i;
              }
            }
            if (tri_ok) {
              auto cdt_faces = local.tri.make_faces();
              auto labels = local.tri.region_labels();
              const bool flip = T2(0) < area2;
              std::size_t fi = 0;
              const auto &kept = local.tri.index_map().kept_ids();
              for (auto ftri : cdt_faces) {
                if ((labels[fi++] & 1) == 0)
                  continue;
                const Index r0 =
                    welded ? local.rep[std::size_t(ftri[0])] : kept[ftri[0]];
                const Index r1 =
                    welded ? local.rep[std::size_t(ftri[1])] : kept[ftri[1]];
                const Index r2 =
                    welded ? local.rep[std::size_t(ftri[2])] : kept[ftri[2]];
                if (r0 == Index(-1) || r1 == Index(-1) || r2 == Index(-1)) {
                  tri_ok = false;
                  break;
                }
                if (flip)
                  local.out.push_back({loop[std::size_t(r0)],
                                       loop[std::size_t(r2)],
                                       loop[std::size_t(r1)]});
                else
                  local.out.push_back({loop[std::size_t(r0)],
                                       loop[std::size_t(r1)],
                                       loop[std::size_t(r2)]});
              }
            }
          }
          if (!tri_ok)
            local.out.erase_till_end(local.out.begin() +
                                     std::ptrdiff_t(before));
          local.fails += Index(!tri_ok);
        }
        local.counts.push_back(Index(local.out.size() - before));
      }
    };
    tf::buffer<Index> counts;
    n_failed = 0;
    auto agg = [&](const local_t &local, const tf::none_t &) {
      tf::core::append(local.out, tris);
      tf::core::append(local.counts, counts);
      n_failed += local.fails;
    };
    tf::blocked_reduce_sequenced_aggregate(
        tf::zip(tf::make_sequence_range(n_loops), descs, loops), tf::none,
        local_t{}, task, agg);
    Index acc = 0;
    for (std::size_t li = 0; li < n_loops; ++li) {
      ranges[li] = {acc, acc + counts[li]};
      acc += counts[li];
    }
    apply_stack_aliases(ag);
  }

  /// @brief Refined mode: quality refinement of the cut surface, computed
  ///        ON the graph. Domains are conserved by construction — the
  ///        classification (loops, connectivity, inclusion) is computed on
  ///        the original arrangement first; the store only changes how the
  ///        already-classified surface is tessellated.
  ///
  /// Pass 1: Ruppert per loop (connectivity splittable mask) -> dyadic
  /// split records + harvested interior points (the refiner's own
  /// placements). Union with bail veto + conforming-face propagation rings.
  /// Pass 2: pure CDT of the union-subdivided boundary + harvested
  /// interiors per loop; conforming uncut faces run frozen quality
  /// insertion (0.3).
  ///
  /// `forms` must carry face_membership + manifold_edge_link policies (the
  /// conforming-ring propagation walks them); the graph's tagged forms do.
  ///
  /// The added points are handed out through `extra_created`
  /// ([splits | steiner], int lattice); the graph appends them to its
  /// unified created-points buffer.
  template <typename AG, typename FC, typename IG, typename Forms,
            typename Converter>
  auto build_refined(const AG &ag, const FC &fc, const IG &ig,
                     const Forms &forms, const Converter &conv,
                     tf::buffer<tf::point<Int, 3>> &extra_created) -> void {
    // The conforming-ring propagation walks loop adjacency; refined
    // mode builds it locally (the arrangement tier carries only the
    // coplanar stacks).
    tf::loop_connectivity<Index> conn_own;
    {
      tf::buffer<Index> point_counts;
      point_counts.allocate(std::size_t(forms.size()));
      for (std::size_t t = 0; t < std::size_t(forms.size()); ++t)
        point_counts[t] = static_cast<Index>(forms[t].points().size());
      conn_own.build(ig, fc, ag.dead_loops(), tf::make_range(point_counts));
    }
    auto conn = conn_own.connectivity_per_face_edge();
    auto ipts = ig.points();
    auto descs = fc.descriptors();
    auto loops = fc.loops();
    const std::size_t n_loops = std::size_t(loops.size());
    n_ig_created = Index(ipts.size());

    auto get_mesh_point = [&](Index tag, Index id) -> tf::point<Int, 3> {
      return conv.convert(
          tf::transformed(forms[std::size_t(tag)].points()[id],
                          tf::frame_of(forms[std::size_t(tag)])));
    };
    auto get_pt3 = [&](Index tag, const auto &v) -> tf::point<Int, 3> {
      return v.source == tf::intersect::graph::vertex_source::original
                 ? get_mesh_point(tag, v.id)
                 : ipts[v.id];
    };
    auto project_setup = [&](Index tag, Index object) {
      auto face = forms[std::size_t(tag)].faces()[object];
      auto a3 = get_mesh_point(tag, Index(face[0]));
      auto b3 = get_mesh_point(tag, Index(face[1]));
      auto c3 = get_mesh_point(tag, Index(face[2]));
      return tf::exact::projection_axes(a3, b3, c3);
    };
    auto face_plane = [&](Index tag, Index object, double nd[3],
                          double ad[3]) {
      auto face = forms[std::size_t(tag)].faces()[object];
      auto fa = get_mesh_point(tag, Index(face[0]));
      auto fb = get_mesh_point(tag, Index(face[1]));
      auto fcp = get_mesh_point(tag, Index(face[2]));
      double ux = double(fb[0]) - fa[0], uy = double(fb[1]) - fa[1],
             uz = double(fb[2]) - fa[2];
      double vx = double(fcp[0]) - fa[0], vy = double(fcp[1]) - fa[1],
             vz = double(fcp[2]) - fa[2];
      nd[0] = uy * vz - uz * vy;
      nd[1] = uz * vx - ux * vz;
      nd[2] = ux * vy - uy * vx;
      ad[0] = double(fa[0]);
      ad[1] = double(fa[1]);
      ad[2] = double(fa[2]);
    };

    // -------------- pass 1: Ruppert discovery + interior harvest (parallel)
    tf::buffer<detail::split_rec<Index>> all_splits;
    tf::buffer<bool> bailed;
    bailed.allocate(n_loops);
    tf::parallel_fill(bailed, false);
    tf::buffer<Index> loop_steiner_off; // n_loops + 1, into steiner_pts
    loop_steiner_off.allocate(n_loops + 1);
    tf::buffer<tf::point<Int, 3>> steiner_pts;
    {
      struct local_t {
        tf::cdt_refiner<Index, Int, Int> rf;
        tf::points_buffer<Int, 2> rpts;
        tf::buffer<Index> flat;
        tf::buffer<bool> splittable;
        tf::buffer<detail::split_rec<Index>> recs;
        tf::buffer<tf::point<Int, 3>> steiner;
        tf::buffer<Index> steiner_counts; // one per visited loop
        tf::buffer<char> bail_flags;      // one per visited loop
        tf::buffer<char> onc, usedp;      // harvest scratch
      };
      std::size_t bailed_fill_pos = 0;
      auto task = [&](auto &&range, local_t &local) {
        for (auto &&[li_z, desc, loop] : range) {
          const std::size_t li = std::size_t(li_z);
          std::size_t steiner_before = local.steiner.size();
          char bail = 0;
          if (desc.tag != Index(-1) && loop.size() >= 3 &&
              !ag.dead_loops()[li]) {
            auto axes = project_setup(desc.tag, desc.object);
            auto conn_row = conn[Index(li)];
            auto &rf = local.rf;
            const std::size_t n = loop.size();
            local.rpts.allocate(n);
            local.flat.clear();
            local.splittable.allocate(n);
            for (std::size_t i = 0; i < n; ++i) {
              auto pt = get_pt3(desc.tag, loop[i]);
              local.rpts[i] =
                  tf::point<Int, 2>{pt[axes.first], pt[axes.second]};
              local.flat.push_back(Index(i));
              local.flat.push_back(Index((i + 1) % n));
              auto nbs = conn_row[i];
              // border edges (no cut neighbour) may split only when whole
              // (both endpoints original): fragment splits are keyed
              // orig-created, which the uncut neighbour's conforming cycle
              // cannot consume yet -- freezing avoids T-junctions there
              const auto &va = loop[i];
              const auto &vb = loop[(i + 1) % n];
              bool whole =
                  va.source ==
                      tf::intersect::graph::vertex_source::original &&
                  vb.source == tf::intersect::graph::vertex_source::original;
              local.splittable[i] =
                  (nbs.size() == 0 && whole) ||
                  (nbs.size() == 1 && descs[nbs[0]].tag == desc.tag);
            }
            tf::cdt_refine_config rc;
            rc.min_quality = 0.3f;
            bool ok = rf.build(local.rpts.points(),
                               tf::make_edges(tf::make_range(local.flat)),
                               tf::make_constant_range(true, n),
                               tf::make_range(local.splittable), rc);
            if (!ok) {
              bail = 1;
            } else {
              auto recs = rf.constraint_splits();
              for (std::size_t j = 0; j < n; ++j) {
                if (recs[j].size() == 0)
                  continue;
                auto ka = detail::pkey(desc.tag, loop[j]);
                auto kb = detail::pkey(desc.tag, loop[(j + 1) % n]);
                bool fwd = ka < kb;
                for (const auto &sr : recs[j]) {
                  std::uint8_t num = sr.numerator;
                  if (!fwd)
                    num = std::uint8_t((1u << sr.depth) - num);
                  local.recs.push_back(
                      {std::min(ka, kb), std::max(ka, kb), num, sr.depth});
                }
              }
              // harvest interiors: used by kept faces, on no constrained edge
              auto rlabels = rf.region_labels();
              auto rp = rf.points();
              local.onc.allocate(rp.size());
              local.usedp.allocate(rp.size());
              std::fill(local.onc.begin(), local.onc.end(), char(0));
              std::fill(local.usedp.begin(), local.usedp.end(), char(0));
              auto &onc = local.onc;
              auto &usedp = local.usedp;
              for (std::size_t f = 0; f < std::size_t(rf.n_faces()); ++f) {
                if (rlabels[f] % 2 != 1)
                  continue;
                auto fc2 = rf.face(Index(f));
                for (int e = 0; e < 3; ++e) {
                  usedp[std::size_t(fc2[e])] = 1;
                  if (rf.constrained(Index(f), e)) {
                    onc[std::size_t(fc2[e])] = 1;
                    onc[std::size_t(fc2[(e + 1) % 3])] = 1;
                  }
                }
              }
              double nd[3], ad[3];
              face_plane(desc.tag, desc.object, nd, ad);
              for (std::size_t i = 0; i < rp.size(); ++i)
                if (usedp[i] && !onc[i])
                  local.steiner.push_back(
                      detail::lift3(rp[i][0], rp[i][1], axes.first,
                                    axes.second, nd, ad));
            }
          }
          local.steiner_counts.push_back(
              Index(local.steiner.size() - steiner_before));
          local.bail_flags.push_back(bail);
        }
      };
      tf::buffer<Index> steiner_counts;
      auto agg = [&](const local_t &local, const tf::none_t &) {
        tf::core::append(local.recs, all_splits);
        tf::core::append(local.steiner, steiner_pts);
        tf::core::append(local.steiner_counts, steiner_counts);
        for (std::size_t i = 0; i < local.bail_flags.size(); ++i)
          bailed[bailed_fill_pos + i] = local.bail_flags[i] != 0;
        bailed_fill_pos += local.bail_flags.size();
      };
      tf::blocked_reduce_sequenced_aggregate(
          tf::zip(tf::make_sequence_range(n_loops), descs, loops), tf::none,
          local_t{}, task, agg);
      loop_steiner_off[0] = 0;
      for (std::size_t li = 0; li < n_loops; ++li)
        loop_steiner_off[li + 1] = loop_steiner_off[li] + steiner_counts[li];
    }

    // --------------------------------------------------- union with veto:
    // bailed loops veto every split on their boundary — their cycle is
    // emitted whole, so consuming a split anywhere else would T-junction
    tf::buffer<std::array<std::array<Index, 2>, 2>> veto;
    for (std::size_t li = 0; li < n_loops; ++li) {
      if (!bailed[li])
        continue;
      auto desc = descs[li];
      auto loop = loops[li];
      for (std::size_t j = 0; j < loop.size(); ++j) {
        auto ka = detail::pkey(desc.tag, loop[j]);
        auto kb = detail::pkey(desc.tag, loop[(j + 1) % loop.size()]);
        veto.push_back({std::min(ka, kb), std::max(ka, kb)});
      }
    }
    tbb::parallel_sort(veto.begin(), veto.end());
    auto vetoed = [&](const std::array<Index, 2> &a,
                      const std::array<Index, 2> &b) {
      std::array<std::array<Index, 2>, 2> k{a, b};
      return std::binary_search(veto.begin(), veto.end(), k);
    };
    tf::buffer<detail::split_rec<Index>> splits;
    tf::buffer<std::array<std::array<Index, 2>, 2>> split_keys;
    tf::buffer<Index> split_off;
    tf::buffer<Index> scan_pos;
    tf::buffer<Index> head_pos;
    // sorted_prefix = length of the already-sorted head; rings append a
    // small tail, so sort the tail and merge instead of re-sorting all
    auto rebuild_union = [&](std::size_t sorted_prefix = 0) {
      if (sorted_prefix == 0) {
        tbb::parallel_sort(all_splits.begin(), all_splits.end());
      } else {
        std::sort(all_splits.begin() + sorted_prefix, all_splits.end());
        std::inplace_merge(all_splits.begin(),
                           all_splits.begin() + sorted_prefix,
                           all_splits.end());
      }
      // dedup + veto + group offsets, two-pass: parallel 0/1 flags, serial
      // exclusive prefix (the flag buffer becomes the position table, so a
      // kept element is one whose position advances), parallel scatter --
      // same arrays the sequential scan produced
      const std::size_t n_rec = all_splits.size();
      scan_pos.clear();
      scan_pos.allocate(n_rec + 1);
      const bool has_veto = veto.size() != 0;
      tf::parallel_for_each(
          tf::make_sequence_range(n_rec), [&](std::size_t i) {
            bool k = (i == 0 || !(all_splits[i] == all_splits[i - 1])) &&
                     (!has_veto ||
                      !vetoed(all_splits[i].ka, all_splits[i].kb));
            scan_pos[i] = Index(k);
          });
      Index n_kept = 0;
      for (std::size_t i = 0; i < n_rec; ++i) {
        Index k = scan_pos[i];
        scan_pos[i] = n_kept;
        n_kept += k;
      }
      scan_pos[n_rec] = n_kept;
      splits.clear();
      splits.allocate(std::size_t(n_kept));
      tf::parallel_for_each(
          tf::make_sequence_range(n_rec), [&](std::size_t i) {
            if (scan_pos[i + 1] != scan_pos[i])
              splits[std::size_t(scan_pos[i])] = all_splits[i];
          });

      const std::size_t n_kept_z = std::size_t(n_kept);
      head_pos.clear();
      head_pos.allocate(n_kept_z + 1);
      tf::parallel_for_each(
          tf::make_sequence_range(n_kept_z), [&](std::size_t j) {
            bool h = j == 0 || splits[j].ka != splits[j - 1].ka ||
                     splits[j].kb != splits[j - 1].kb;
            head_pos[j] = Index(h);
          });
      Index n_keys = 0;
      for (std::size_t j = 0; j < n_kept_z; ++j) {
        Index h = head_pos[j];
        head_pos[j] = n_keys;
        n_keys += h;
      }
      head_pos[n_kept_z] = n_keys;
      split_keys.clear();
      split_keys.allocate(std::size_t(n_keys));
      split_off.clear();
      split_off.allocate(std::size_t(n_keys) + 1);
      tf::parallel_for_each(
          tf::make_sequence_range(n_kept_z), [&](std::size_t j) {
            if (head_pos[j + 1] != head_pos[j]) {
              split_keys[std::size_t(head_pos[j])] = {splits[j].ka,
                                                      splits[j].kb};
              split_off[std::size_t(head_pos[j])] = Index(j);
            }
          });
      split_off[std::size_t(n_keys)] = n_kept;
    };
    rebuild_union();
    auto splits_of = [&](const std::array<Index, 2> &a,
                         const std::array<Index, 2> &b)
        -> std::pair<Index, Index> {
      std::array<std::array<Index, 2>, 2> k{std::min(a, b), std::max(a, b)};
      auto it = std::lower_bound(split_keys.begin(), split_keys.end(), k);
      if (it == split_keys.end() || !((*it)[0] == k[0] && (*it)[1] == k[1]))
        return {0, 0};
      auto idx = it - split_keys.begin();
      return {split_off[std::size_t(idx)], split_off[std::size_t(idx) + 1]};
    };

    // ------------------ conforming discovery + propagation rings (serial):
    // uncut faces receiving splits on a shared original edge demand splits
    // on their OTHER edges, graded by cyclic Lipschitz sizing of their
    // (subdivided) boundary; deepen-only rings cascade until stable
    const std::size_t n_tags = forms.size();
    tf::core::std_vector<tf::buffer<bool>> cut_mask(n_tags);
    for (std::size_t t = 0; t < n_tags; ++t) {
      cut_mask[t].allocate(forms[t].size());
      tf::parallel_fill(cut_mask[t], false);
    }
    for (std::size_t li = 0; li < n_loops; ++li)
      if (descs[li].tag != Index(-1))
        cut_mask[std::size_t(descs[li].tag)][std::size_t(descs[li].object)] =
            true;

    tf::buffer<std::array<Index, 2>> conf;
    tf::buffer<std::array<std::array<Index, 2>, 2>> ring_keys;
    // keys == nullptr: full scan of the union. keys != nullptr: only faces
    // adjacent to those keys can have changed since the last ring -- their
    // records are a superset refresh, the union dedups the rest
    auto discover_conf =
        [&](const tf::buffer<std::array<std::array<Index, 2>, 2>> *keys =
                nullptr) {
      conf.clear();
      const auto &scan = keys ? *keys : split_keys;
      auto emit_key = [&](const std::array<std::array<Index, 2>, 2> &key,
                          tf::buffer<std::array<Index, 2>> &out) {
        const auto &ka = key[0];
        const auto &kb = key[1];
        if (ka[0] == Index(-1) || kb[0] == Index(-1))
          return; // created endpoint: not an original mesh edge
        Index tag = ka[0];
        if (kb[0] != tag)
          return;
        Index u = ka[1], v = kb[1];
        const auto &form = forms[std::size_t(tag)];
        for (auto f_id : form.face_membership()[u]) {
          if (cut_mask[std::size_t(tag)][std::size_t(f_id)])
            continue;
          auto face = form.faces()[f_id];
          for (int e = 0; e < 3; ++e)
            if ((Index(face[e]) == u && Index(face[(e + 1) % 3]) == v) ||
                (Index(face[e]) == v && Index(face[(e + 1) % 3]) == u)) {
              out.push_back({tag, Index(f_id)});
              break;
            }
        }
      };
      // the sort+unique canonicalizes order, so candidates can be found in
      // parallel; ring scans are tiny and stay serial
      if (scan.size() < 512) {
        for (std::size_t k = 0; k < scan.size(); ++k)
          emit_key(scan[k], conf);
      } else {
        struct disc_local_t {
          tf::buffer<std::array<Index, 2>> found;
        };
        tf::blocked_reduce_sequenced_aggregate(
            tf::make_range(scan), tf::none, disc_local_t{},
            [&](auto &&range, disc_local_t &local) {
              for (const auto &key : range)
                emit_key(key, local.found);
            },
            [&](const disc_local_t &local, const tf::none_t &) {
              tf::core::append(local.found, conf);
            });
      }
      tbb::parallel_sort(conf.begin(), conf.end());
      conf.erase_till_end(std::unique(conf.begin(), conf.end()));
    };

    {
      // rings run sequentially (deepen-only convergence), but WITHIN a
      // ring every conforming face emits its demands independently --
      // parallel with block-local scratch, ordered aggregation keeps
      // the record stream deterministic
      struct ring_local_t {
        tf::buffer<std::array<double, 2>> fpts;
        tf::buffer<double> fh, felen;
        tf::buffer<std::size_t> fedge;
        tf::buffer<std::uint32_t> fparams;
        tf::buffer<detail::split_rec<Index>> recs;
      };
      for (int ring = 0; ring < 4; ++ring) {
        discover_conf(ring == 0 ? nullptr : &ring_keys);
        std::size_t before = all_splits.size();
        auto ring_task = [&](auto &&range, ring_local_t &local) {
          for (const auto &key : range) {
            Index tag = key[0], f_id = key[1];
            auto face = forms[std::size_t(tag)].faces()[f_id];
            auto axes = project_setup(tag, f_id);
            local.fpts.clear();
            local.fedge.clear();
            local.fparams.clear();
            for (int e = 0; e < 3; ++e) {
              Index u = Index(face[e]), v = Index(face[(e + 1) % 3]);
              auto pu = get_mesh_point(tag, u);
              auto pv = get_mesh_point(tag, v);
              std::array<double, 2> xu{double(pu[axes.first]),
                                       double(pu[axes.second])};
              std::array<double, 2> xv{double(pv[axes.first]),
                                       double(pv[axes.second])};
              local.fpts.push_back(xu);
              local.fedge.push_back(std::size_t(e));
              local.fparams.push_back(0);
              auto ka = detail::pkey_orig(tag, u);
              auto kb = detail::pkey_orig(tag, v);
              auto [s0, s1] = splits_of(ka, kb);
              bool fwd = ka < kb;
              for (Index k = 0; k < s1 - s0; ++k) {
                Index idx = fwd ? s0 + k : s1 - 1 - k;
                std::uint32_t p = splits[std::size_t(idx)].param();
                if (!fwd)
                  p = 256u - p;
                double t = double(p) / 256.0;
                local.fpts.push_back({xu[0] + t * (xv[0] - xu[0]),
                                      xu[1] + t * (xv[1] - xu[1])});
                local.fedge.push_back(std::size_t(e));
                local.fparams.push_back(p);
              }
            }
            const std::size_t m = local.fpts.size();
            detail::cycle_sizing(local.fpts, false, local.fh, local.felen);
            auto no_lfs = [](const std::array<double, 2> &) {
              return 1e300;
            };
            for (std::size_t i = 0; i < m; ++i) {
              std::size_t in = (i + 1) % m;
              std::size_t e = local.fedge[i];
              auto link_row =
                  forms[std::size_t(tag)].manifold_edge_link()[f_id];
              if (!link_row[e].is_simple())
                continue;
              std::uint32_t pa = local.fparams[i];
              std::uint32_t pb =
                  (in != 0 && local.fedge[in] == e) ? local.fparams[in]
                                                    : 256u;
              Index u = Index(face[e]), v = Index(face[(e + 1) % 3]);
              auto ka = detail::pkey_orig(tag, u);
              auto kb = detail::pkey_orig(tag, v);
              bool fwd = ka < kb;
              detail::dyadic_split(
                  pa, pb, local.fpts[i], local.fpts[in], local.fh[i],
                  local.fh[in],
                  [&](std::uint32_t p) {
                    std::uint32_t pc = fwd ? p : 256u - p;
                    auto [num, depth] = detail::param_to_rec(pc);
                    local.recs.push_back(
                        {std::min(ka, kb), std::max(ka, kb), num, depth});
                  },
                  no_lfs);
            }
          }
        };
        auto ring_agg = [&](const ring_local_t &local, const tf::none_t &) {
          tf::core::append(local.recs, all_splits);
        };
        tf::blocked_reduce_sequenced_aggregate(tf::make_range(conf), tf::none,
                                               ring_local_t{}, ring_task,
                                               ring_agg);
        if (all_splits.size() == before)
          break;
        ring_keys.clear();
        for (std::size_t i = before; i < all_splits.size(); ++i)
          ring_keys.push_back({all_splits[i].ka, all_splits[i].kb});
        std::sort(ring_keys.begin(), ring_keys.end());
        ring_keys.erase_till_end(
            std::unique(ring_keys.begin(), ring_keys.end()));
        rebuild_union(before);
      }
      discover_conf();
    }

    // ----------------------------------- split point materialization: 3D
    // int dyadic interpolation, computed ONCE per unique split so every
    // consumer emits the identical lattice point (watertight by ids)
    auto key_pt3 = [&](const std::array<Index, 2> &k) -> tf::point<Int, 3> {
      if (k[0] == Index(-1))
        return ipts[k[1]];
      return get_mesh_point(k[0], k[1]);
    };
    tf::buffer<tf::point<Int, 3>> split_pts;
    tf::buffer<std::array<std::array<Index, 2>, 2>> split_parents;
    split_pts.allocate(splits.size());
    split_parents.allocate(splits.size());
    tf::parallel_for_each(
        tf::make_sequence_range(splits.size()), [&](std::size_t i) {
          auto a = key_pt3(splits[i].ka);
          auto b = key_pt3(splits[i].kb);
          double t = double(splits[i].num) / double(1u << splits[i].depth);
          split_pts[i] = tf::point<Int, 3>{
              Int(std::llround(double(a[0]) + t * (double(b[0]) - a[0]))),
              Int(std::llround(double(a[1]) + t * (double(b[1]) - a[1]))),
              Int(std::llround(double(a[2]) + t * (double(b[2]) - a[2])))};
          split_parents[i] = {splits[i].ka, splits[i].kb};
        });

    // ----------------------- pass 2: loops (parallel, pure CDT, ordered)
    tf::buffer<Index> loop_tri_offsets; // n_loops + 1
    loop_tri_offsets.allocate(n_loops + 1);
    tf::buffer<std::array<Index, 3>> loop_tris_enc;
    {
      struct local_t {
        tf::cdt_refiner<Index, Int, Int> rf;
        tf::points_buffer<Int, 2> rpts;
        tf::buffer<Index> flat;
        tf::buffer<tf::point<Int, 3>> lp3;
        tf::buffer<Index> lref;
        tf::buffer<std::array<Index, 3>> tris;
        tf::buffer<Index> tri_counts;
        detail::emit_scratch<Index, Int> sc;
        std::size_t fans = 0;
      };
      auto task = [&](auto &&range, local_t &local) {
        for (auto &&[li_z, desc, loop] : range) {
          const std::size_t li = std::size_t(li_z);
          std::size_t before = local.tris.size();
          if (desc.tag != Index(-1) && loop.size() >= 3 &&
              !ag.dead_loops()[li]) {
            auto axes = project_setup(desc.tag, desc.object);
            local.lp3.clear();
            local.lref.clear();
            for (std::size_t i = 0; i < loop.size(); ++i) {
              local.lp3.push_back(get_pt3(desc.tag, loop[i]));
              local.lref.push_back(Index(i));
              auto ka = detail::pkey(desc.tag, loop[i]);
              auto kb =
                  detail::pkey(desc.tag, loop[(i + 1) % loop.size()]);
              auto [s0, s1] = splits_of(ka, kb);
              bool fwd = ka < kb;
              for (Index k = 0; k < s1 - s0; ++k) {
                Index idx = fwd ? s0 + k : s1 - 1 - k;
                local.lp3.push_back(split_pts[std::size_t(idx)]);
                local.lref.push_back(detail::split_ref(idx));
              }
            }
            std::size_t nc = local.lp3.size();
            for (Index sid = loop_steiner_off[li];
                 sid < loop_steiner_off[li + 1]; ++sid) {
              local.lp3.push_back(steiner_pts[std::size_t(sid)]);
              local.lref.push_back(detail::steiner_ref(sid));
            }
            double nd[3], ad[3];
            face_plane(desc.tag, desc.object, nd, ad);
            detail::cdt_emit(local.rf, local.rpts, local.flat, local.lp3,
                             local.lref, nc, axes.first, axes.second, nd, ad,
                             0.0f, local.tris,
                             static_cast<tf::buffer<tf::point<Int, 3>> *>(
                                 nullptr),
                             local.fans, local.sc);
          }
          local.tri_counts.push_back(Index(local.tris.size() - before));
        }
      };
      tf::buffer<Index> tri_counts;
      auto agg = [&](const local_t &local, const tf::none_t &) {
        tf::core::append(local.tris, loop_tris_enc);
        tf::core::append(local.tri_counts, tri_counts);
      };
      tf::blocked_reduce_sequenced_aggregate(
          tf::zip(tf::make_sequence_range(n_loops), descs, loops), tf::none,
          local_t{}, task, agg);
      loop_tri_offsets[0] = 0;
      for (std::size_t li = 0; li < n_loops; ++li)
        loop_tri_offsets[li + 1] = loop_tri_offsets[li] + tri_counts[li];
    }

    // ---- pass 2: conforming faces (parallel, frozen 0.3, steiner rebase)
    tf::buffer<std::array<Index, 3>> conf_tris_enc;
    conf_tri_offsets.allocate(conf.size() + 1);
    {
      struct local_t {
        tf::cdt_refiner<Index, Int, Int> rf;
        tf::points_buffer<Int, 2> rpts;
        tf::buffer<Index> flat;
        tf::buffer<tf::point<Int, 3>> lp3;
        tf::buffer<Index> lref;
        tf::buffer<std::array<Index, 3>> tris;
        tf::buffer<Index> tri_counts;
        tf::buffer<tf::point<Int, 3>> steiner;
        detail::emit_scratch<Index, Int> sc;
        std::size_t fans = 0;
      };
      auto task = [&](auto &&range, local_t &local) {
        for (auto &&key : range) {
          Index tag = key[0], f_id = key[1];
          std::size_t before = local.tris.size();
          auto face = forms[std::size_t(tag)].faces()[f_id];
          auto axes = project_setup(tag, f_id);
          local.lp3.clear();
          local.lref.clear();
          for (int e = 0; e < 3; ++e) {
            Index u = Index(face[e]), v = Index(face[(e + 1) % 3]);
            local.lp3.push_back(get_mesh_point(tag, u));
            local.lref.push_back(Index(e));
            auto ka = detail::pkey_orig(tag, u);
            auto kb = detail::pkey_orig(tag, v);
            auto [s0, s1] = splits_of(ka, kb);
            bool fwd = ka < kb;
            for (Index k = 0; k < s1 - s0; ++k) {
              Index idx = fwd ? s0 + k : s1 - 1 - k;
              local.lp3.push_back(split_pts[std::size_t(idx)]);
              local.lref.push_back(detail::split_ref(idx));
            }
          }
          double nd[3], ad[3];
          face_plane(tag, f_id, nd, ad);
          detail::cdt_emit(local.rf, local.rpts, local.flat, local.lp3,
                           local.lref, local.lp3.size(), axes.first,
                           axes.second, nd, ad, 0.3f, local.tris,
                           &local.steiner, local.fans, local.sc);
          local.tri_counts.push_back(Index(local.tris.size() - before));
        }
      };
      tf::buffer<Index> tri_counts;
      auto agg = [&](const local_t &local, const tf::none_t &) {
        // rebase this chunk's LOCAL steiner refs to global steiner ids
        Index base = Index(steiner_pts.size());
        auto old = conf_tris_enc.size();
        tf::core::append(local.tris, conf_tris_enc);
        for (std::size_t i = old; i < conf_tris_enc.size(); ++i)
          for (int c2 = 0; c2 < 3; ++c2) {
            Index v = conf_tris_enc[i][std::size_t(c2)];
            if (v < 0) {
              Index e = -v - 1;
              if (e % 2 == 1)
                conf_tris_enc[i][std::size_t(c2)] =
                    detail::steiner_ref((e - 1) / 2 + base);
            }
          }
        tf::core::append(local.steiner, steiner_pts);
        tf::core::append(local.tri_counts, tri_counts);
      };
      tf::blocked_reduce_sequenced_aggregate(tf::make_range(conf), tf::none,
                                             local_t{}, task, agg);
      conf_tri_offsets[0] = 0;
      for (std::size_t c = 0; c < conf.size(); ++c)
        conf_tri_offsets[c + 1] = conf_tri_offsets[c] + tri_counts[c];
    }

    // ------------------------------------- conversion to vertex storage
    const Index n_splits = Index(splits.size());
    // extra created points handed to the graph: [splits | steiner]
    extra_created.allocate(split_pts.size() + steiner_pts.size());
    extra_kind.allocate(extra_created.size());
    extra_parent.allocate(extra_created.size());
    tf::parallel_copy(tf::make_range(split_pts),
                      tf::make_range(extra_created.begin(),
                                     extra_created.begin() +
                                         split_pts.size()));
    tf::parallel_copy(tf::make_range(steiner_pts),
                      tf::make_range(extra_created.begin() +
                                         split_pts.size(),
                                     extra_created.end()));
    tf::parallel_for_each(tf::make_sequence_range(split_pts.size()),
                          [&](std::size_t i) {
                            extra_kind[i] = 0;
                            extra_parent[i] = split_parents[i];
                          });
    tf::parallel_for_each(
        tf::make_sequence_range(n_loops), [&](std::size_t li) {
          auto desc = descs[li];
          for (Index sid = loop_steiner_off[li];
               sid < loop_steiner_off[li + 1]; ++sid) {
            extra_kind[std::size_t(n_splits + sid)] = 1;
            extra_parent[std::size_t(n_splits + sid)] = {
                std::array<Index, 2>{desc.tag, desc.object},
                std::array<Index, 2>{Index(-1), Index(-1)}};
          }
        });
    // conforming-face steiner parents (ids beyond the loop steiner) are
    // filled during the conforming-stream conversion below

    auto neg_vertex = [&](Index r) -> vertex_t {
      Index e = -r - 1;
      if (e % 2 == 0) // split
        return vertex_t{tf::intersect::graph::vertex_source::created,
                        n_ig_created + e / 2,
                        tf::topo_id<short>{0, tf::topo_type::edge}};
      return vertex_t{tf::intersect::graph::vertex_source::created,
                      n_ig_created + n_splits + (e - 1) / 2,
                      tf::topo_id<short>{0, tf::topo_type::face}};
    };
    auto to_vertex = [&](Index r, const auto &loop) -> vertex_t {
      return r >= 0 ? loop[r] : neg_vertex(r);
    };

    ranges.allocate(n_loops);
    rev.allocate(n_loops);
    tris.allocate(loop_tris_enc.size());
    tf::parallel_for_each(
        tf::make_sequence_range(n_loops), [&](std::size_t li) {
          ranges[li] = {loop_tri_offsets[li], loop_tri_offsets[li + 1]};
          auto loop = loops[li];
          for (Index j = loop_tri_offsets[li]; j < loop_tri_offsets[li + 1];
               ++j) {
            auto t = loop_tris_enc[std::size_t(j)];
            tris[std::size_t(j)] = {to_vertex(t[0], loop),
                                    to_vertex(t[1], loop),
                                    to_vertex(t[2], loop)};
          }
        });
    apply_stack_aliases(ag);

    // conforming stream: corners are original vertices of the face
    conf_keys = std::move(conf);
    conf_tris.allocate(conf_tris_enc.size());
    tf::parallel_for_each(
        tf::make_sequence_range(conf_keys.size()), [&](std::size_t c) {
          Index tag = conf_keys[c][0], f_id = conf_keys[c][1];
          auto face = forms[std::size_t(tag)].faces()[f_id];
          auto corner = [&](Index r) -> vertex_t {
            if (r >= 0)
              return vertex_t{tf::intersect::graph::vertex_source::original,
                              Index(face[r]),
                              tf::topo_id<short>{short(r),
                                                 tf::topo_type::vertex}};
            return neg_vertex(r);
          };
          for (Index j = conf_tri_offsets[c]; j < conf_tri_offsets[c + 1];
               ++j) {
            auto t = conf_tris_enc[std::size_t(j)];
            conf_tris[std::size_t(j)] = {corner(t[0]), corner(t[1]),
                                         corner(t[2])};
            for (int k = 0; k < 3; ++k) { // steiner provenance = this face
              Index r = t[std::size_t(k)];
              if (r < 0 && (-r - 1) % 2 == 1) {
                std::size_t sid = std::size_t(n_splits + ((-r - 1) - 1) / 2);
                extra_kind[sid] = 1;
                extra_parent[sid] = {std::array<Index, 2>{tag, f_id},
                                     std::array<Index, 2>{Index(-1),
                                                          Index(-1)}};
              }
            }
          }
        });
  }
};

} // namespace tf::cut

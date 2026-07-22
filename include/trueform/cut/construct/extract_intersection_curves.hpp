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

#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/buffer.hpp"
#include "../../core/curves_buffer.hpp"
#include "../../core/edges.hpp"
#include "../../core/point.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/points.hpp"
#include "../../core/views/zip.hpp"
#include "../../topology/connect_edges_to_paths.hpp"
#include "tbb/parallel_sort.h"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/none.hpp"
#include "../face_regions.hpp"
#include "../loop_connectivity.hpp"
#include "../make_coplanar_loop_pairs.hpp"
#include <algorithm>
#include <utility>
#include <array>
#include <type_traits>

namespace tf::cut {

/// Curves from seam edges over a created-points table, with only the
/// REFERENCED points copied to the output: the same sentinel-discovery
/// map the mesh constructs use, so a retired or unreferenced created
/// id never becomes a stray output vertex.
template <typename RealOut, typename Index, typename Edges,
          typename CreatedPoints, typename Converter>
auto curves_from_seam_edges(Edges &edges, const CreatedPoints &created_pts,
                            const Converter &conv)
    -> tf::curves_buffer<Index, RealOut, 3> {
  tf::buffer<Index> point_map;
  point_map.allocate(created_pts.size());
  const Index sentinel = static_cast<Index>(created_pts.size());
  tf::parallel_fill(point_map, sentinel);
  tf::buffer<Index> used;
  for (auto &e : edges)
    for (int c = 0; c < 2; ++c)
      if (point_map[std::size_t(e[std::size_t(c)])] == sentinel) {
        point_map[std::size_t(e[std::size_t(c)])] = Index(used.size());
        used.push_back(e[std::size_t(c)]);
      }
  for (auto &e : edges) {
    e[0] = point_map[std::size_t(e[0])];
    e[1] = point_map[std::size_t(e[1])];
  }
  tf::curves_buffer<Index, RealOut, 3> cb;
  cb.paths_buffer() =
      tf::connect_edges_to_paths(tf::make_edges(tf::make_range(edges)));
  cb.points_buffer().allocate(used.size());
  auto gathered =
      tf::make_indirect_range(tf::make_range(used), created_pts);
  if constexpr (std::is_integral_v<RealOut>) {
    tf::parallel_copy(tf::make_points(gathered), cb.points());
  } else {
    tf::parallel_copy(
        tf::make_points(tf::make_mapped_range(
            gathered,
            [&conv](const auto &pt) { return conv.deconvert(pt); })),
        cb.points());
  }
  return cb;
}

/// The intersection-curve network of an arrangement at REGION grain.
/// A seam is a region walk edge (boundary or hole) with a neighbour of
/// a different form tag; within a single tag, where tags cannot speak,
/// it is a non-manifold edge (2+ surviving neighbour walks); and in
/// both cases the contact border of a coincident overlap — where a
/// coplanar-stacked region meets an unpaired one — is a seam (the
/// interior of the stack stays silent, its duplicate walks are already
/// collapsed out of `conn`). Endpoints are created-vertex ids into the
/// unified created-points table; a weld that retired a created seam
/// endpoint in favour of an original identity is keyed back through
/// its smallest retired created twin, so seam keying never leaves
/// created-id space. Each edge is emitted only from its smallest
/// incident region, so the set is duplicate-free. `rt` supplies only
/// the weld merges — pass @c tf::none when no triangulation ran
/// (curves-only reads): the intersection graph shares vertices by
/// construction, so there is nothing to resolve.
template <typename RealOut, typename Index, typename Rt, typename Int,
          typename Conn, typename Stacked, typename CreatedPoints,
          typename Converter>
auto extract_intersection_curves(const tf::face_regions<Index, Int> &fr,
                                 const Rt &rt, const Conn &conn,
                                 const Stacked &stacked,
                                 const CreatedPoints &created_pts,
                                 const Converter &conv)
    -> tf::curves_buffer<Index, RealOut, 3> {
  auto loops = fr.loops();
  auto holes = fr.holes();
  auto loop_holes = fr.loop_holes();
  auto descs = fr.descriptors();
  const Index n_tags = static_cast<Index>(fr.tag_offsets().size()) - 1;

  auto endpoint = [&](Index tag, const auto &v) -> Index {
    if constexpr (std::is_same_v<Rt, tf::none_t>) {
      (void)tag;
      if (v.source == tf::intersect::graph::vertex_source::created)
        return v.id;
      return Index(-1);
    } else {
      // walks carry pre-weld identities; two incident regions may
      // hold different members of a welded pair, so seams key through
      // the survivor
      auto rv = rt.resolve(tag, v);
      if (rv.source == tf::intersect::graph::vertex_source::created)
        return rv.id;
      Index best = Index(-1);
      const std::array<Index, 2> key{tag, rv.id};
      for (const auto &m : rt.merges())
        if (m.to_key == key && m.from[0] == n_tags &&
            (best == Index(-1) || m.from[1] < best))
          best = m.from[1];
      return best;
    }
  };

  // Within a single tag the seam rule is non-manifold incidence, and
  // the collapsed connectivity cannot answer it: its rows at
  // non-manifold edges are the region-formation PAIRING, not an
  // incidence list. The incidence is latent in the walks — sort the
  // walk edges by identity key; the group is the incidence, its
  // smallest loop the emitter.
  auto akey = [&rt, n_tags](Index tag, const auto &v) -> std::array<Index, 2> {
    if constexpr (std::is_same_v<Rt, tf::none_t>) {
      (void)rt;
      if (v.source == tf::intersect::graph::vertex_source::created)
        return {n_tags, v.id};
      return {tag, v.id};
    } else {
      // welded identities must land in ONE occurrence group
      auto rv = rt.resolve(tag, v);
      if (rv.source == tf::intersect::graph::vertex_source::created)
        return {n_tags, rv.id};
      return {tag, rv.id};
    }
  };
  tf::buffer<std::array<Index, 5>> nm_keys;
  if (n_tags == Index(1)) {
    tf::generic_generate(
        tf::enumerate(tf::zip(loops, conn)), nm_keys,
        [&](const auto &item, auto &out) {
          auto &&[li_z, tup] = item;
          auto &&[boundary_b, edge_neighbors_b] = tup;
          const auto &boundary = boundary_b;
          if (edge_neighbors_b.size() == 0)
            return;
          const Index li = static_cast<Index>(li_z);
          const Index my_tag = descs[li].tag;
          auto scan = [&](const auto &walk) {
            const Index n = static_cast<Index>(walk.size());
            for (Index j = 0; j < n; ++j) {
              auto ka = akey(my_tag, walk[j]);
              auto kb = akey(my_tag, walk[(j + 1) % n]);
              if (kb < ka)
                std::swap(ka, kb);
              out.push_back({ka[0], ka[1], kb[0], kb[1], li});
            }
          };
          scan(boundary);
          for (auto h : loop_holes[li])
            scan(holes[h]);
        });
    tbb::parallel_sort(nm_keys.begin(), nm_keys.end());
  }
  auto nm_group = [&](const std::array<Index, 4> &k)
      -> std::pair<std::size_t, Index> {
    auto lo = std::lower_bound(
        nm_keys.begin(), nm_keys.end(), k,
        [](const std::array<Index, 5> &a, const std::array<Index, 4> &b) {
          return std::array<Index, 4>{a[0], a[1], a[2], a[3]} < b;
        });
    std::size_t cnt = 0;
    Index min_loop = Index(0);
    for (auto it = lo; it != nm_keys.end() &&
                       std::array<Index, 4>{(*it)[0], (*it)[1], (*it)[2],
                                            (*it)[3]} == k;
         ++it, ++cnt)
      if (cnt == 0 || (*it)[4] < min_loop)
        min_loop = (*it)[4];
    return {cnt, min_loop};
  };

  tf::buffer<std::array<Index, 2>> ie;
  tf::generic_generate(
      tf::enumerate(tf::zip(loops, conn)), ie,
      [&](const auto &item, auto &out) {
        auto &&[li_z, tup] = item;
        auto &&[boundary_b, edge_neighbors_b] = tup;
        const auto &boundary = boundary_b;
        const auto edge_neighbors = edge_neighbors_b;
        // Dead (coplanar-duplicate) regions keep their vertices but
        // have empty cleaned connectivity — skip them.
        if (edge_neighbors.size() == 0)
          return;
        const Index li = static_cast<Index>(li_z);
        const Index my_tag = descs[li].tag;
        // Walk slots follow compute_face_link_per_edge: within a walk
        // slot j = (v[j], v[(j+1) % n]); walks pack boundary-first.
        const bool self_stacked =
            stacked.size() != 0 && bool(stacked[std::size_t(li)]);
        Index slot = 0;
        auto scan = [&](const auto &walk) {
          const Index n = static_cast<Index>(walk.size());
          for (Index j = 0; j < n; ++j, ++slot) {
            bool cross = false, is_min = true;
            if (n_tags == Index(1)) {
              auto ka = akey(my_tag, walk[j]);
              auto kb = akey(my_tag, walk[(j + 1) % n]);
              if (kb < ka)
                std::swap(ka, kb);
              auto [cnt, min_loop] =
                  nm_group({ka[0], ka[1], kb[0], kb[1]});
              cross = cnt >= 3;
              is_min = li == min_loop;
            }
            for (auto nj : edge_neighbors[slot]) {
              if (descs[nj].tag != my_tag)
                cross = true;
              if (stacked.size() != 0 &&
                  bool(stacked[std::size_t(nj)]) != self_stacked)
                cross = true;
              if (n_tags != Index(1) && nj < li)
                is_min = false; // a smaller region on this edge emits it
            }
            if (cross && is_min) {
              const Index a = endpoint(my_tag, walk[j]);
              const Index b = endpoint(my_tag, walk[(j + 1) % n]);
              if (std::make_signed_t<Index>(a) < 0 ||
                  std::make_signed_t<Index>(b) < 0 || a == b)
                continue;
              if constexpr (std::is_same_v<Rt, tf::none_t>) {
                out.push_back({std::min(a, b), std::max(a, b)});
              } else {
                // the seam is a carrier like any other: traverse the
                // split sub-chain the mesh emitted, whatever produced
                // the splits (quality, rings, crossing recovery)
                Index prev = a;
                rt.for_each_edge_split(
                    my_tag, walk[j], walk[(j + 1) % n],
                    [&](const auto &sv) {
                      const auto rv = rt.resolve(my_tag, sv);
                      if (rv.source !=
                          tf::intersect::graph::vertex_source::created)
                        return;
                      if (rv.id != prev)
                        out.push_back(
                            {std::min(prev, rv.id), std::max(prev, rv.id)});
                      prev = rv.id;
                    });
                if (prev != b)
                  out.push_back({std::min(prev, b), std::max(prev, b)});
              }
            }
          }
        };
        scan(boundary);
        for (auto h : loop_holes[li])
          scan(holes[h]);
      });

  // ie arrives completion-ordered (generic_generate): sort to a
  // canonical total order so point numbering and paths are
  // byte-deterministic
  tbb::parallel_sort(ie.begin(), ie.end());
  return curves_from_seam_edges<RealOut, Index>(ie, created_pts, conv);
}

/// Curves straight from an intersection graph — no triangulation, no
/// arrangement graph: the region structure, its coplanar collapse, and
/// the collapsed connectivity are all a curve read needs.
template <typename RealOut, typename Index, typename Int,
          typename ApplyToFace, typename GetMeshPoint, typename PointCounts,
          typename Converter>
auto make_region_curves(const tf::intersection_graph<Index, Int> &ig,
                        const ApplyToFace &apply_to_face,
                        const GetMeshPoint &get_mesh_point,
                        const PointCounts &point_counts, const Converter &conv)
    -> tf::curves_buffer<Index, RealOut, 3> {
  tf::face_regions<Index, Int> fr;
  fr.build(ig, apply_to_face, get_mesh_point);
  auto pairs = make_coplanar_loop_pairs_all(fr);
  tf::buffer<char> dead;
  dead.allocate(fr.loops().size());
  tf::parallel_fill(dead, char(0));
  tf::buffer<char> stacked;
  if (pairs.size() != 0) {
    stacked.allocate(fr.loops().size());
    tf::parallel_fill(stacked, char(0));
    for (const auto &cp : pairs) {
      dead[std::size_t(cp.loop_b)] = char(1);
      stacked[std::size_t(cp.loop_a)] = char(1);
      stacked[std::size_t(cp.loop_b)] = char(1);
    }
  }
  tf::loop_connectivity<Index> conn;
  conn.build(fr, tf::make_range(dead), point_counts,
             static_cast<Index>(ig.points().size()));
  return extract_intersection_curves<RealOut, Index>(
      fr, tf::none, conn.connectivity_per_face_edge(), tf::make_range(stacked),
      ig.points(), conv);
}

} // namespace tf::cut

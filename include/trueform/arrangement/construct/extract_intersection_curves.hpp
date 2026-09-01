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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../../core/curves_buffer.hpp"
#include "../../core/edges.hpp"
#include "../../core/range.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/points.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../topology/connect_edges_to_paths.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tf::arrangement {

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

/// The intersection-curve network of an arrangement at TRIANGLE grain.
/// A seam is a constraint edge of the exposed triangle stream between
/// two created vertices whose incidence group carries two form tags, or
/// mixes a coincident-stack triangle with an unstacked one (the
/// same-tag stack border). Filler (non-constraint) edges never make
/// topology, and the stream already speaks split pieces, so an edge
/// needs no sub-chain traversal. Dead (coplanar-duplicate) triangles do
/// not exist to this read, which IS the coincident-overlap rule: an
/// overlap-interior edge is carried by the survivor alone and stays
/// silent, while the contact border is also carried by both tags'
/// regular triangles outside the stack and speaks through the cross-tag
/// test.
///
/// Within a single tag no cross-tag test can speak, so the seam rule is
/// non-manifold incidence: the group sweep already counts a constraint
/// edge's live incidences, and three or more make it a seam. Multi-tag
/// arrangements never use the count — same-tag non-manifold edges there
/// are folds, not seams.
///
/// `triangles` are the exposed stream's vertex triples, `constraint_bits`
/// its per-triangle mask (bit c marks the edge c -> c + 1), and `tags`,
/// `dead`, and `stacked` its per-triangle facts. Endpoints are created
/// vertex ids into the unified created-points table.
template <typename RealOut, typename Index, typename Triangles,
          typename ConstraintBits, typename Tags, typename Dead,
          typename Stacked, typename CreatedPoints, typename Converter>
auto extract_triangle_seam_curves(const Triangles &triangles,
                                  const ConstraintBits &constraint_bits,
                                  const Tags &tags, const Dead &dead,
                                  const Stacked &stacked, Index n_tags,
                                  const CreatedPoints &created_pts,
                                  const Converter &conv)
    -> tf::curves_buffer<Index, RealOut, 3> {
  // {a, b, tag, stacked} per constraint edge between created vertices
  tf::buffer<std::array<Index, 4>> recs;
  tf::generic_generate(
      tf::make_sequence_range(Index(triangles.size())), recs,
      [&](Index t, auto &out) {
        if (dead[std::size_t(t)])
          return;
        const auto &tv = triangles[std::size_t(t)];
        for (int c = 0; c < 3; ++c) {
          if (!(constraint_bits[std::size_t(t)] & std::uint8_t(1 << c)))
            continue;
          const auto &a = tv[std::size_t(c)];
          const auto &b = tv[std::size_t((c + 1) % 3)];
          if (a.source != tf::intersect::graph::vertex_source::created ||
              b.source != tf::intersect::graph::vertex_source::created ||
              a.id == b.id)
            continue;
          out.push_back({std::min(a.id, b.id), std::max(a.id, b.id),
                         tags[std::size_t(t)],
                         Index(stacked[std::size_t(t)])});
        }
      });
  tbb::parallel_sort(recs.begin(), recs.end());

  tf::buffer<std::array<Index, 2>> seams;
  const bool single_tag = n_tags == Index(1);
  std::size_t i = 0;
  while (i < recs.size()) {
    std::size_t j = i;
    bool cross = false;
    for (; j < recs.size() && recs[j][0] == recs[i][0] &&
           recs[j][1] == recs[i][1];
         ++j)
      cross |= recs[j][2] != recs[i][2] || recs[j][3] != recs[i][3];
    if (single_tag && j - i >= 3)
      cross = true;
    if (cross)
      seams.push_back({recs[i][0], recs[i][1]});
    i = j;
  }
  return curves_from_seam_edges<RealOut, Index>(seams, created_pts, conv);
}

} // namespace tf::arrangement

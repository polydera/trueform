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

#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/memory.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../face_regions.hpp"
#include "./consolidate_region_merges.hpp"
#include "./make_region_interior_points.hpp"
#include "./splice_region_triangulations.hpp"
#include "./triangulate_region.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace tf::cut {

/// @ingroup cut
/// @brief The wave's last word: triangulate the regions still refusing
///        after it, resolving.
///
/// A region that refuses when the wave has nothing left to broadcast is
/// asking for a split that does not exist at this precision — the crossing
/// falls on a vertex that is already there. Resolving completes that build,
/// the created crossing points identify with the vertices their parameters
/// name, and adding no split is not a failure but the answer. Asking
/// preserved again would refuse forever. Identifications reported as merges
/// reach the other carriers through the weld conformance that follows.
template <typename Index, typename Int, typename ApplyToFace,
          typename GetMeshPoint>
auto resolve_refusing_regions(
    const tf::face_regions<Index, Int> &regions,
    const tf::intersection_graph<Index, Int> &intersection_graph,
    const ApplyToFace &apply_to_face, const GetMeshPoint &get_mesh_point,
    Index n_tags, Index n_intersection_points,
    tf::buffer<tf::point<Int, 3>> &extra_points,
    const tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>>
        &splits,
    tf::buffer<tf::cut::detail::region_triangulation_merge<Index>> &merges,
    tf::buffer<std::array<Index, 2>> &ranges,
    tf::buffer<std::array<tf::intersect::graph::vertex<Index>, 3>> &triangles,
    tf::buffer<Index> &failed,
    const tf::offset_block_buffer<Index, Index> &steiners) -> void {
  if (failed.size() == 0)
    return;
  auto descriptors = regions.descriptors();
  auto loops = regions.loops();
  auto loop_holes = regions.loop_holes();
  auto intersection_points = intersection_graph.points();

  struct local_t {
    tf::cut::detail::region_triangulation_workspace<Index, Int> workspace;
    tf::buffer<std::array<tf::intersect::graph::vertex<Index>, 3>> triangles;
    tf::buffer<Index> counts;
    tf::buffer<Index> failed;
  };
  tf::buffer<std::array<tf::intersect::graph::vertex<Index>, 3>>
      resolved_triangles;
  tf::buffer<Index> resolved_counts;
  tf::buffer<Index> still_failed;
  tf::buffer<tf::cut::detail::region_triangulation_merge<Index>>
      resolve_merges;
  auto resolve = [&](auto &&range, local_t &local) {
    auto apply_to_face_copy = apply_to_face;
    auto get_mesh_point_copy = get_mesh_point;
    for (auto loop_index : range) {
      const std::size_t before = local.triangles.size();
      const bool ok = tf::cut::triangulate_region(
          regions, descriptors[loop_index], loops[loop_index],
          loop_holes[loop_index], apply_to_face_copy, get_mesh_point_copy,
          intersection_points, n_tags, n_intersection_points, extra_points,
          splits, merges, local.workspace, local.triangles, true,
          tf::cut::detail::region_build::resolve,
          tf::cut::make_region_interior_points<Index, Int>(
              loop_index, steiners, n_intersection_points, extra_points));
      if (!ok) {
        local.triangles.erase_till_end(local.triangles.begin() +
                                       std::ptrdiff_t(before));
        local.failed.push_back(loop_index);
      }
      local.counts.push_back(Index(local.triangles.size() - before));
    }
  };
  auto aggregate_resolve = [&](const local_t &local, const tf::none_t &) {
    tf::core::append(local.triangles, resolved_triangles);
    tf::core::append(local.counts, resolved_counts);
    tf::core::append(local.failed, still_failed);
    tf::core::append(local.workspace.merges, resolve_merges);
  };
  tf::blocked_reduce_sequenced_aggregate(tf::make_range(failed), tf::none,
                                         local_t{}, resolve,
                                         aggregate_resolve);

  tf::buffer<std::array<Index, 2>> resolved_ranges;
  resolved_ranges.allocate(failed.size());
  Index resolved_offset = 0;
  for (std::size_t index = 0; index < failed.size(); ++index) {
    resolved_ranges[index] = {resolved_offset,
                              resolved_offset + resolved_counts[index]};
    resolved_offset += resolved_counts[index];
  }
  tf::cut::splice_region_triangulations(failed, resolved_ranges,
                                        resolved_triangles, ranges, triangles);
  failed = std::move(still_failed);
  tf::cut::consolidate_region_merges<Index>(merges, resolve_merges);
}

} // namespace tf::cut

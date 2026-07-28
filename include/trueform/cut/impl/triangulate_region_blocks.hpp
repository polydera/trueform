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
#include "../../core/point.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../face_regions.hpp"
#include "./make_region_interior_points.hpp"
#include "./triangulate_region.hpp"
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int, typename ApplyToFace,
          typename GetMeshPoint, typename IntersectionPoints,
          typename DeadLoops>
auto triangulate_region_blocks(
    const tf::face_regions<Index, Int> &regions,
    const ApplyToFace &apply_to_face,
    const GetMeshPoint &get_mesh_point,
    const IntersectionPoints &intersection_points, Index n_tags,
    Index n_intersection_points,
    const tf::buffer<tf::point<Int, 3>> &extra_points,
    const tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>>
        &splits,
    const tf::buffer<tf::cut::detail::region_triangulation_merge<Index>>
        &merges,
    const DeadLoops &dead_loops,
    tf::buffer<std::array<Index, 2>> &ranges,
    tf::buffer<
        std::array<tf::intersect::graph::vertex<Index>, 3>>
        &triangles,
    tf::buffer<Index> &failed,
    tf::buffer<
        tf::cut::detail::region_triangulation_merge<Index>>
        &pending_merges,
    const tf::offset_block_buffer<Index, Index> &steiners = {}) -> void {
  struct local {
    tf::cut::detail::region_triangulation_workspace<Index, Int> workspace;
    tf::buffer<
        std::array<tf::intersect::graph::vertex<Index>, 3>>
        triangles;
    tf::buffer<Index> counts;
    tf::buffer<Index> failed;
  };

  auto interior_of = [&](Index region) {
    return tf::cut::make_region_interior_points<Index, Int>(
        region, steiners, n_intersection_points, extra_points);
  };
  auto descriptors = regions.descriptors();
  auto loops = regions.loops();
  auto loop_holes = regions.loop_holes();
  ranges.clear();
  ranges.allocate(std::size_t(loops.size()));
  triangles.clear();
  failed.clear();
  pending_merges.clear();
  auto task = [&](auto &&range, local &state) {
    auto apply_to_face_copy = apply_to_face;
    auto get_mesh_point_copy = get_mesh_point;
    for (auto &&[loop_index, descriptor, loop, hole_ids] : range) {
      const std::size_t before = state.triangles.size();
      bool ok = true;
      if (descriptor.tag != Index(-1) && loop.size() >= 3 &&
          !(dead_loops.size() != 0 &&
            bool(dead_loops[std::size_t(loop_index)]))) {
        ok = tf::cut::triangulate_region(
            regions, descriptor, loop, hole_ids, apply_to_face_copy,
            get_mesh_point_copy, intersection_points, n_tags,
            n_intersection_points, extra_points, splits, merges,
            state.workspace, state.triangles, splits.size() != 0,
            tf::cut::detail::region_build::preserve,
            interior_of(Index(loop_index)));
      }
      if (!ok) {
        state.triangles.erase_till_end(
            state.triangles.begin() + std::ptrdiff_t(before));
        state.failed.push_back(Index(loop_index));
      }
      state.counts.push_back(
          Index(state.triangles.size() - before));
    }
  };
  tf::buffer<Index> counts;
  auto aggregate = [&](const local &state, const tf::none_t &) {
    tf::core::append(state.triangles, triangles);
    tf::core::append(state.counts, counts);
    tf::core::append(state.failed, failed);
    tf::core::append(state.workspace.merges, pending_merges);
  };
  tf::blocked_reduce_sequenced_aggregate(
      tf::zip(tf::make_sequence_range(std::size_t(loops.size())),
              descriptors, loops, loop_holes),
      tf::none, local{}, task, aggregate);

  Index offset = 0;
  for (std::size_t loop = 0; loop < std::size_t(loops.size()); ++loop) {
    ranges[loop] = {offset, offset + counts[loop]};
    offset += counts[loop];
  }
}

} // namespace tf::cut

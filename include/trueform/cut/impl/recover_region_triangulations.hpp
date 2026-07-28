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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/memory.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_identity.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../face_regions.hpp"
#include "./consolidate_region_merges.hpp"
#include "./discover_region_crossings.hpp"
#include "./make_region_interior_points.hpp"
#include "./materialize_region_crossings.hpp"
#include "./region_triangulation_point.hpp"
#include "./resolve_refusing_regions.hpp"
#include "./resolve_region_vertex.hpp"
#include "./sort_region_splits.hpp"
#include "./splice_region_triangulations.hpp"
#include "./triangulate_region.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::cut {

template <typename Index, typename Int, typename ApplyToFace,
          typename GetMeshPoint, typename DeadLoops>
auto recover_region_triangulations(
    const tf::face_regions<Index, Int> &regions,
    const tf::intersection_graph<Index, Int> &intersection_graph,
    const ApplyToFace &apply_to_face,
    const GetMeshPoint &get_mesh_point, const DeadLoops &dead_loops,
    Index n_tags, Index n_intersection_points,
    tf::buffer<tf::point<Int, 3>> &extra_points,
    tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>> &splits,
    tf::buffer<tf::cut::detail::region_triangulation_merge<Index>> &merges,
    tf::buffer<std::array<Index, 2>> &ranges,
    tf::buffer<
        std::array<tf::intersect::graph::vertex<Index>, 3>> &triangles,
    tf::buffer<Index> &failed,
    const tf::offset_block_buffer<Index, Index> &steiners = {}) -> void {
  auto interior_of = [&](Index region) {
    return tf::cut::make_region_interior_points<Index, Int>(
        region, steiners, n_intersection_points, extra_points);
  };
  // Promotion reports a failed face as its slot past the region ranges. The
  // wave's carriers — descriptors, walks, holes, the Steiner table — are
  // region-indexed, so a promoted slot has no entry in any of them and is
  // carried through untouched rather than looked up.
  tf::buffer<Index> promoted_failed;
  {
    std::size_t write = 0;
    for (std::size_t index = 0; index < failed.size(); ++index) {
      if (std::size_t(failed[index]) < std::size_t(regions.loops().size()))
        failed[write++] = failed[index];
      else
        promoted_failed.push_back(failed[index]);
    }
    failed.erase_till_end(failed.begin() + std::ptrdiff_t(write));
  }
  auto descriptors = regions.descriptors();
  auto loops = regions.loops();
  auto loop_holes = regions.loop_holes();
  auto holes = regions.holes();
  auto intersection_points = intersection_graph.points();
  auto get_point =
      [&](Index tag,
          const tf::intersect::graph::vertex<Index> &vertex) {
    return tf::cut::region_triangulation_point(
        n_intersection_points, extra_points, tag, vertex,
        intersection_points, get_mesh_point);
  };
  auto touches_merge = [&](Index tag, const auto &walk) {
    for (const auto &vertex : walk)
      if (tf::cut::resolve_region_vertex_key(
              n_tags, tag, vertex, merges) !=
          tf::cut::detail::region_vertex_key(n_tags, tag, vertex))
        return true;
    return false;
  };
  bool merges_fresh = merges.size() != 0;

  for (int round = 0;
       round < tf::cut::detail::region_triangulation_max_rounds &&
       (failed.size() || merges_fresh);
       ++round) {
    const std::size_t split_count = splits.size();
    struct discovery_local {
      tf::cut::detail::region_triangulation_workspace<Index, Int> workspace;
      tf::buffer<
          std::array<tf::intersect::graph::vertex<Index>, 3>>
          scratch_triangles;
      tf::buffer<tf::cut::detail::region_crossing_proposal<Index>>
          proposals;
      tf::buffer<
          std::array<tf::intersect::graph::vertex<Index>, 2>>
          parent_edges;
      tf::buffer<typename tf::exact::meta<Int>::param_type> parameters;
    };
    tf::buffer<tf::cut::detail::region_crossing_proposal<Index>>
        proposals;
    tf::buffer<std::array<tf::intersect::graph::vertex<Index>, 2>>
        parent_edges;
    tf::buffer<typename tf::exact::meta<Int>::param_type> parameters;
    tf::buffer<tf::cut::detail::region_triangulation_merge<Index>>
        pending_merges;
    auto discover = [&](auto &&range, discovery_local &local) {
      auto apply_to_face_copy = apply_to_face;
      auto get_mesh_point_copy = get_mesh_point;
      for (auto loop_index : range) {
        const auto &descriptor = descriptors[loop_index];
        local.scratch_triangles.clear();
        tf::cut::triangulate_region(
            regions, descriptor, loops[loop_index],
            loop_holes[loop_index], apply_to_face_copy,
            get_mesh_point_copy, intersection_points, n_tags,
            n_intersection_points, extra_points, splits, merges,
            local.workspace, local.scratch_triangles, true,
            tf::cut::detail::region_build::preserve,
            interior_of(loop_index));
        tf::cut::discover_region_crossings<Index, Int>(
            local.workspace, n_tags, descriptor.tag, local.proposals,
            local.parent_edges, local.parameters);
      }
    };
    auto aggregate_discovery =
        [&](const discovery_local &local, const tf::none_t &) {
      tf::core::append(local.proposals, proposals);
      tf::core::append(local.parent_edges, parent_edges);
      tf::core::append(local.parameters, parameters);
      tf::core::append(local.workspace.merges, pending_merges);
    };
    tf::blocked_reduce_sequenced_aggregate(
        tf::make_range(failed), tf::none, discovery_local{}, discover,
        aggregate_discovery);

    tf::cut::materialize_region_crossings<Index, Int>(
        n_tags, n_intersection_points, proposals, parent_edges, parameters,
        get_point, extra_points, splits);
    // A weld the discovery pass found is as fresh as one the retry found:
    // it has to reach the touched-scan below, or the other carriers of that
    // vertex are never told. Dropping this answer also makes the weld
    // permanently invisible — the round-end consolidation sees it as a
    // duplicate and reports no change.
    merges_fresh =
        tf::cut::consolidate_region_merges<Index>(merges, pending_merges) ||
        merges_fresh;
    if (splits.size() == split_count && !merges_fresh)
      break;
    tf::cut::sort_region_splits(splits);

    tf::buffer<char> touched;
    touched.allocate(std::size_t(loops.size()));
    tf::parallel_for_each(
        tf::make_sequence_range(Index(loops.size())), [&](Index loop_index) {
          touched[std::size_t(loop_index)] = char(0);
          if (descriptors[loop_index].tag == Index(-1) ||
              loops[loop_index].size() < 3 ||
              (dead_loops.size() != 0 &&
               bool(dead_loops[std::size_t(loop_index)])))
            return;
          bool hit = false;
          const Index tag = descriptors[loop_index].tag;
          auto scan = [&](const auto &walk) {
            const auto size = walk.size();
            for (std::size_t i = 0, j = size - 1;
                 i < size && !hit; j = i++) {
              const auto range =
                  tf::cut::detail::find_region_edge_splits(
                      tf::cut::detail::region_edge_key(
                          tf::cut::detail::region_vertex_key(
                              n_tags, tag, walk[j]),
                          tf::cut::detail::region_vertex_key(
                              n_tags, tag, walk[i])),
                      splits);
              hit = range.first != range.second;
            }
          };
          scan(loops[loop_index]);
          for (auto hole : loop_holes[loop_index]) {
            if (hit)
              break;
            scan(holes[hole]);
          }
          if (!hit && merges_fresh) {
            hit = touches_merge(tag, loops[loop_index]);
            for (auto hole : loop_holes[loop_index]) {
              if (hit)
                break;
              hit = touches_merge(tag, holes[hole]);
            }
          }
          touched[std::size_t(loop_index)] = char(hit);
        });
    tf::buffer<Index> dirty;
    for (Index loop = 0; loop < Index(loops.size()); ++loop)
      if (touched[std::size_t(loop)])
        dirty.push_back(loop);
    for (auto loop : failed)
      dirty.push_back(loop);
    std::sort(dirty.begin(), dirty.end());
    dirty.erase_till_end(std::unique(dirty.begin(), dirty.end()));

    struct retry_local {
      tf::cut::detail::region_triangulation_workspace<Index, Int> workspace;
      tf::buffer<
          std::array<tf::intersect::graph::vertex<Index>, 3>>
          triangles;
      tf::buffer<Index> counts;
      tf::buffer<Index> failed;
    };
    tf::buffer<
        std::array<tf::intersect::graph::vertex<Index>, 3>>
        replacement_triangles;
    tf::buffer<Index> replacement_counts;
    tf::buffer<Index> remaining_failed;
    tf::buffer<tf::cut::detail::region_triangulation_merge<Index>>
        retry_merges;
    auto retry = [&](auto &&range, retry_local &local) {
      auto apply_to_face_copy = apply_to_face;
      auto get_mesh_point_copy = get_mesh_point;
      for (auto loop_index : range) {
        const std::size_t before = local.triangles.size();
        const bool ok = tf::cut::triangulate_region(
            regions, descriptors[loop_index], loops[loop_index],
            loop_holes[loop_index], apply_to_face_copy,
            get_mesh_point_copy, intersection_points, n_tags,
            n_intersection_points, extra_points, splits, merges,
            local.workspace, local.triangles, true,
            tf::cut::detail::region_build::preserve,
            interior_of(loop_index));
        if (!ok) {
          local.triangles.erase_till_end(
              local.triangles.begin() + std::ptrdiff_t(before));
          local.failed.push_back(loop_index);
        }
        local.counts.push_back(
            Index(local.triangles.size() - before));
      }
    };
    auto aggregate_retry = [&](const retry_local &local,
                               const tf::none_t &) {
      tf::core::append(local.triangles, replacement_triangles);
      tf::core::append(local.counts, replacement_counts);
      tf::core::append(local.failed, remaining_failed);
      tf::core::append(local.workspace.merges, retry_merges);
    };
    tf::blocked_reduce_sequenced_aggregate(
        tf::make_range(dirty), tf::none, retry_local{}, retry,
        aggregate_retry);

    tf::buffer<std::array<Index, 2>> replacement_ranges;
    replacement_ranges.allocate(dirty.size());
    Index offset = 0;
    for (std::size_t index = 0; index < dirty.size(); ++index) {
      replacement_ranges[index] = {
          offset, offset + replacement_counts[index]};
      offset += replacement_counts[index];
    }
    tf::cut::splice_region_triangulations(
        dirty, replacement_ranges, replacement_triangles, ranges,
        triangles);
    failed = std::move(remaining_failed);
    merges_fresh =
        tf::cut::consolidate_region_merges<Index>(merges, retry_merges);
  }

  tf::cut::resolve_refusing_regions(
      regions, intersection_graph, apply_to_face, get_mesh_point, n_tags,
      n_intersection_points, extra_points, splits, merges, ranges, triangles,
      failed, steiners);
  tf::core::append(promoted_failed, failed);
}

} // namespace tf::cut

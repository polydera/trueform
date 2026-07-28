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
#include "../../exact/rebase_parameter.hpp"
#include "./constraint_parent_span.hpp"

#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/edges.hpp"
#include "../../core/memory.hpp"
#include "../../core/none.hpp"
#include "../../core/point.hpp"
#include "../../core/range.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../core/views/constant.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../topology/cdt_refine_config.hpp"
#include "../../topology/cdt_refiner.hpp"
#include "../detail/region_triangulation_identity.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../face_regions.hpp"
#include "./collect_region_steiner_points.hpp"
#include "./discover_region_crossings.hpp"
#include "./inject_region_steiners.hpp"
#include "./loop_connectivity.hpp"
#include "./materialize_region_crossings.hpp"
#include "./region_refinement_fingerprint.hpp"
#include "./region_triangulation_point.hpp"
#include "./sort_region_splits.hpp"
#include "./triangulate_region.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::cut {

template <typename Index, typename Int, typename ApplyToForm,
          typename ApplyToFace, typename GetMeshPoint,
          typename IntersectionPoints, typename DeadLoops>
auto discover_region_refinement(
    const tf::face_regions<Index, Int> &regions, Index n_tags,
    const ApplyToForm &apply_to_form,
    const ApplyToFace &apply_to_face,
    const GetMeshPoint &get_mesh_point,
    const IntersectionPoints &intersection_points,
    const tf::cdt_refine_config &config, const DeadLoops &dead_loops,
    Index n_intersection_points,
    const tf::buffer<std::array<Index, 2>> &region_fingerprints,
    const tf::cdt_refine_config &refined_config,
    const tf::offset_block_buffer<Index, Index> &persisted_steiners,
    const tf::buffer<tf::cut::detail::region_triangulation_merge<Index>>
        &merges,
    tf::buffer<tf::point<Int, 3>> &extra_points,
    tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>>
        &splits) -> tf::cut::detail::region_refinement_discovery<Index, Int> {
  auto descriptors = regions.descriptors();
  auto loops = regions.loops();
  auto loop_holes = regions.loop_holes();

  tf::cut::loop_connectivity<Index> connectivity_owner;
  {
    tf::buffer<Index> point_counts;
    point_counts.allocate(std::size_t(n_tags));
    for (Index tag = 0; tag < n_tags; ++tag)
      apply_to_form(tag, [&](const auto &form) {
        point_counts[std::size_t(tag)] =
            static_cast<Index>(form.points().size());
      });
    connectivity_owner.build(
        regions, dead_loops, tf::make_range(point_counts),
        Index(std::size_t(n_intersection_points) + extra_points.size()));
  }
  auto connectivity =
      connectivity_owner.connectivity_per_carrier_edge();

  tf::buffer<std::array<std::array<Index, 2>, 2>> seam_keys;
  tf::generic_generate(
      tf::enumerate(loops), seam_keys,
      [&](const auto &indexed_loop, auto &output) {
        auto &&[loop_index, loop] = indexed_loop;
        const Index region = Index(loop_index);
        const auto descriptor = descriptors[region];
        if (descriptor.tag == Index(-1) || loop.size() < 3 ||
            (dead_loops.size() != 0 &&
             bool(dead_loops[std::size_t(loop_index)])))
          return;
        auto scan = [&](const auto &walk) {
          const std::size_t size = walk.size();
          for (std::size_t i = 0, j = size - 1; i < size; j = i++)
            output.push_back(tf::cut::detail::region_edge_key(
                tf::cut::detail::region_vertex_key(
                    n_tags, descriptor.tag, walk[j]),
                tf::cut::detail::region_vertex_key(
                    n_tags, descriptor.tag, walk[i])));
        };
        scan(loop);
        for (auto hole : loop_holes[region])
          scan(regions.holes()[hole]);
      });
  tbb::parallel_sort(seam_keys.begin(), seam_keys.end());
  auto is_seam =
      [&](const std::array<std::array<Index, 2>, 2> &key) {
    const auto range =
        std::equal_range(seam_keys.begin(), seam_keys.end(), key);
    return range.second - range.first >= 3;
  };

  struct local {
    tf::cut::detail::region_triangulation_workspace<Index, Int> workspace;
    tf::cdt_refiner<Index, Int, Int> refiner;
    tf::buffer<bool> splittable;
    tf::buffer<
        std::array<tf::intersect::graph::vertex<Index>, 3>>
        scratch_triangles;
    tf::buffer<tf::cut::detail::region_split_proposal<Index, Int>> proposals;
    tf::buffer<tf::point<Int, 3>> steiner_points;
    tf::buffer<std::array<Index, 2>> steiner_counts;
    tf::buffer<char> point_flags;
    tf::buffer<Index> failed;
    tf::buffer<tf::cut::detail::region_crossing_proposal<Index>>
        crossing_proposals;
    tf::buffer<typename tf::exact::meta<Int>::param_type> crossing_parameters;
    tf::buffer<std::array<tf::intersect::graph::vertex<Index>, 2>>
        crossing_parents;
  };
  tf::cut::detail::region_refinement_discovery<Index, Int> out;
  tf::buffer<Index> failed;
  tf::buffer<tf::cut::detail::region_crossing_proposal<Index>>
      crossing_proposals;
  tf::buffer<typename tf::exact::meta<Int>::param_type> crossing_parameters;
  tf::buffer<std::array<tf::intersect::graph::vertex<Index>, 2>>
      crossing_parents;
  auto task = [&](auto &&range, local &state) {
    auto apply_to_face_copy = apply_to_face;
    auto get_mesh_point_copy = get_mesh_point;
    for (auto &&[loop_index, descriptor, loop, hole_ids] : range) {
      if (descriptor.tag == Index(-1) || loop.size() < 3 ||
          (dead_loops.size() != 0 &&
           bool(dead_loops[std::size_t(loop_index)])))
        continue;
      // A region is already refined only when the WHOLE request is already
      // satisfied. split_encroached decides whether boundary constraints
      // split at all, so a run that turns it on asks a different question
      // of every region than the run before it, at any quality.
      if (region_fingerprints.size() != 0 &&
          config.min_quality <= refined_config.min_quality &&
          config.split_encroached <= refined_config.split_encroached &&
          region_fingerprints[std::size_t(loop_index)] ==
              tf::cut::region_refinement_fingerprint(
                  regions, Index(loop_index), n_tags, splits,
                  persisted_steiners))
        continue;

      state.scratch_triangles.clear();
      if (!tf::cut::triangulate_region(
              regions, descriptor, loop, hole_ids, apply_to_face_copy,
              get_mesh_point_copy, intersection_points, n_tags,
              n_intersection_points, extra_points, splits, merges,
              state.workspace, state.scratch_triangles, true,
              tf::cut::detail::region_build::assemble)) {
        state.failed.push_back(Index(loop_index));
        continue;
      }
      tf::cut::inject_region_steiners(
          Index(loop_index), persisted_steiners, n_intersection_points,
          extra_points, state.workspace);
      auto edges = tf::make_edges(tf::make_blocked_range<2>(
          tf::make_range(state.workspace.cons.data(),
                         state.workspace.cons.size())));
      state.splittable.allocate(edges.size());
      {
        auto row = connectivity[Index(loop_index)];
        for (std::size_t edge = 0;
             edge < state.workspace.cons_verts.size(); ++edge) {
          auto neighbours = row[state.workspace.cons_slot[edge]];
          const auto &carrier =
              bool(state.workspace.cons_sub[edge])
                  ? state.workspace.cons_parent[edge]
                  : state.workspace.cons_verts[edge];
          const bool whole =
              carrier[0].source ==
                  tf::intersect::graph::vertex_source::original &&
              carrier[1].source ==
                  tf::intersect::graph::vertex_source::original;
          state.splittable[edge] =
              ((neighbours.size() == 0 && whole) ||
               (neighbours.size() == 1 &&
                descriptors[neighbours[0]].tag == descriptor.tag)) &&
              !is_seam(tf::cut::detail::region_edge_key(
                  tf::cut::detail::region_vertex_key(
                      n_tags, descriptor.tag,
                      state.workspace.cons_verts[edge][0]),
                  tf::cut::detail::region_vertex_key(
                      n_tags, descriptor.tag,
                      state.workspace.cons_verts[edge][1])));
        }
      }
      state.refiner.clear();
      if (!state.refiner.build(
              tf::make_points(state.workspace.pts), edges,
              tf::make_constant_range(true, edges.size()),
              tf::make_range(state.splittable), config)) {
        state.failed.push_back(Index(loop_index));
        tf::cut::discover_region_crossings<Index, Int>(
            state.workspace, n_tags, descriptor.tag,
            state.crossing_proposals, state.crossing_parents,
            state.crossing_parameters);
        continue;
      }

      const auto constraint_splits = state.refiner.constraint_splits();
      const typename tf::exact::meta<Int>::param_type maximum_parameter =
          typename tf::exact::meta<Int>::param_type(1)
          << tf::exact::meta<Int>::split_grid_bits;
      for (std::size_t edge = 0;
           edge < state.workspace.cons_verts.size(); ++edge) {
        if (constraint_splits[edge].size() == 0)
          continue;
        using param_t = typename tf::exact::meta<Int>::param_type;
        const auto span = tf::cut::constraint_parent_span<Index, Int>(
            state.workspace, n_tags, descriptor.tag, edge);
        const auto &parent = span.edge;
        const param_t a_parameter = span.t0;
        const param_t b_parameter = span.t1;
        for (const auto &split : constraint_splits[edge]) {
          using T2 = typename tf::exact::meta<Int>::T2;
          const param_t parameter = tf::exact::rebase_parameter<Int>(
              param_t(split.numerator), a_parameter, b_parameter,
              T2(1) << split.depth);
          if (parameter <= param_t(0) || maximum_parameter <= parameter)
            continue;
          state.proposals.push_back({parent, descriptor.tag, parameter});
        }
      }

      const std::size_t steiner_before = state.steiner_points.size();
      tf::cut::collect_region_steiner_points(
          state.workspace, state.refiner, state.point_flags,
          state.steiner_points);
      // Every region this pass looked at, including the ones it chose no
      // point for. A region absent from this list was skipped as already
      // refined, and keeps the interior points it was given then.
      state.steiner_counts.push_back(
          {Index(loop_index),
           Index(state.steiner_points.size() - steiner_before)});
    }
  };
  auto aggregate = [&](const local &state, const tf::none_t &) {
    tf::core::append(state.proposals, out.proposals);
    tf::core::append(state.steiner_points, out.steiner_points);
    tf::core::append(state.steiner_counts, out.steiner_counts);
    tf::core::append(state.failed, failed);
    tf::core::append(state.crossing_proposals, crossing_proposals);
    tf::core::append(state.crossing_parameters, crossing_parameters);
    tf::core::append(state.crossing_parents, crossing_parents);
  };
  tf::blocked_reduce_sequenced_aggregate(
      tf::zip(tf::make_sequence_range(std::size_t(loops.size())),
              descriptors, loops, loop_holes),
      tf::none, local{}, task, aggregate);

  out.refused = Index(failed.size());
  if (failed.size() != 0) {
    const std::size_t steiner_count_before = out.steiner_counts.size();
    auto point =
        [&](Index tag,
            const tf::intersect::graph::vertex<Index> &vertex) {
      return tf::cut::region_triangulation_point(
          n_intersection_points, extra_points, tag, vertex,
          intersection_points, get_mesh_point);
    };
    for (int round = 0;
         round < tf::cut::detail::region_triangulation_max_rounds &&
         failed.size() != 0;
         ++round) {
      const std::size_t split_count = splits.size();
      tf::cut::materialize_region_crossings<Index, Int>(
          n_tags, n_intersection_points, crossing_proposals,
          crossing_parents, crossing_parameters, point, extra_points, splits);
      if (splits.size() == split_count)
        break;
      tf::cut::sort_region_splits(splits);
      crossing_proposals.clear();
      crossing_parameters.clear();
      crossing_parents.clear();
      tf::buffer<Index> retry;
      std::swap(retry, failed);
      local retry_local{};
      task(tf::zip(
               tf::make_range(retry),
               tf::make_indirect_range(tf::make_range(retry),
                                       descriptors),
               tf::make_indirect_range(tf::make_range(retry), loops),
               tf::make_indirect_range(tf::make_range(retry),
                                       loop_holes)),
           retry_local);
      aggregate(retry_local, tf::none);
      out.recovered +=
          Index(retry.size()) - Index(retry_local.failed.size());
    }

    if (out.steiner_counts.size() != steiner_count_before) {
      tf::buffer<std::array<Index, 3>> order;
      order.allocate(out.steiner_counts.size());
      Index offset = 0;
      for (std::size_t index = 0; index < out.steiner_counts.size();
           ++index) {
        order[index] = {out.steiner_counts[index][0], offset,
                        out.steiner_counts[index][1]};
        offset += out.steiner_counts[index][1];
      }
      std::sort(order.begin(), order.end());
      tf::buffer<tf::point<Int, 3>> ordered_points;
      ordered_points.allocate(out.steiner_points.size());
      Index write = 0;
      for (std::size_t index = 0; index < order.size(); ++index) {
        out.steiner_counts[index] = {order[index][0], order[index][2]};
        for (Index point_index = 0; point_index < order[index][2];
             ++point_index)
          ordered_points[std::size_t(write++)] =
              out.steiner_points
                  [std::size_t(order[index][1] + point_index)];
      }
      out.steiner_points = std::move(ordered_points);
    }
  }
  return out;
}

} // namespace tf::cut

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
#include "../../core/checked.hpp"
#include "../../core/edges.hpp"
#include "../../core/none.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/plane_frame.hpp"
#include "../../exact/snap_parameter_to_edge.hpp"
#include "../../exact/vertex.hpp"
#include "../../topology/cdt/constrained_delaunay_refinement_producer.hpp"
#include "../../topology/cdt_refine_config.hpp"
#include "./collect_plane_flat_welds.hpp"
#include "./plane_flat_weld.hpp"
#include "./plane_member_statements.hpp"
#include "./plane_recovery_birth.hpp"
#include "./plane_refinement_plan.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>

namespace tf::arrangement {

/// The reusable input scratch filled for one plane refinement job.
template <typename Index, typename Int> struct plane_refinement_workspace {
  tf::buffer<Index> ends;
  tf::points_buffer<Int, 2> pts2;
  tf::buffer<Index> cons;
  tf::buffer<char> bnd;
  tf::buffer<Index> cons_group;
  tf::buffer<std::array<Index, 2>> cons_row;
  tf::buffer<tf::arrangement::plane_member_statement<Index>> cons_statements;
  tf::buffer<short> cons_side;
  tf::buffer<std::array<Index, 2>> member_ticket;
  int face_orientation = 1;
};

/// Discover refinement choices on explicit plane jobs without changing the
/// arrangement that prepared them.
///
/// `prepare(job, workspace)` fills key-ordered constraints and their current
/// group ids. `group_ends(job, group)` returns that current piece's endpoint
/// flats in its DEFINITION's own order — the frame the producer measures its
/// choices in — and `group_endpoints(job, group)` returns the matching exact
/// 3D endpoints in that same order. The name is stated here, from that pair
/// ascending, together with the inversion the two orders state.
/// `frame_of(job)` returns the owning plane frame and `source_of(job, group)`
/// names the table holding the current group. Steiner block `p` belongs to
/// global plane `p` in `[0, plane_extent)`.
template <typename Index, typename Int, typename Jobs, typename Prepare,
          typename GroupEnds, typename GroupEndpoints, typename FrameOf,
          typename SourceOf>
auto discover_plane_refinement(const Jobs &jobs, Index plane_extent,
                               Index n_flat, const Prepare &prepare,
                               const GroupEnds &group_ends,
                               const GroupEndpoints &group_endpoints,
                               const FrameOf &frame_of,
                               const SourceOf &source_of,
                               const tf::cdt_refine_config &config)
    -> tf::arrangement::plane_refinement_plan<Index, Int> {
  using param_t = typename tf::exact::meta<Int>::param_type;

  struct local_t {
    tf::arrangement::plane_refinement_workspace<Index, Int> workspace;
    tf::topology::cdt::constrained_delaunay_refinement_producer<Index, Int, Int>
        producer;
    tf::buffer<bool> splittable;
    tf::buffer<tf::arrangement::plane_refinement_split<Index, Int>> splits;
    tf::buffer<std::array<Index, 3>> steiner_runs;
    tf::buffer<tf::exact::pt3<Int>> steiner_points;
    tf::buffer<Index> representatives;
    tf::buffer<tf::arrangement::plane_flat_weld<Index>> welds;
    tf::buffer<Index> refused_planes;
  };

  tf::arrangement::plane_refinement_plan<Index, Int> plan;
  tf::buffer<std::array<Index, 4>> steiner_runs;
  tf::buffer<tf::exact::pt3<Int>> steiner_points;

  auto task = [&](auto &&range, local_t &local) {
    for (const auto &job : range) {
      auto &workspace = local.workspace;
      if (!prepare(job, workspace)) {
        local.refused_planes.push_back(job.plane);
        continue;
      }

      local.splittable.allocate(workspace.bnd.size());
      for (std::size_t c = 0; c < workspace.bnd.size(); ++c)
        local.splittable[c] = bool(workspace.bnd[c]);

      const auto built =
          local.producer.build(workspace.pts2.points(),
                               tf::make_edges(tf::make_blocked_range<2>(
                                   tf::make_range(workspace.cons))),
                               tf::make_range(workspace.bnd),
                               tf::make_range(local.splittable), config);
      const auto &index_map = local.producer.index_map();
      // The seed CDT cannot create vertices. Equal input/output counts prove
      // the dominant no-weld case without a scan.
      const auto weld_map_valid =
          local.producer.has_prepared_input_index_map() &&
          (index_map.kept_ids().size() == workspace.ends.size() ||
           tf::arrangement::collect_plane_flat_welds(
               tf::make_range(workspace.ends), index_map.f(),
               index_map.kept_ids(), Index(index_map.f().size()),
               Index(index_map.kept_ids().size()), n_flat, job.plane,
               local.representatives, local.welds));
      if (!built) {
        local.refused_planes.push_back(job.plane);
        continue;
      }
      if (!weld_map_valid) {
        local.refused_planes.push_back(job.plane);
        continue;
      }

      const auto constraint_splits = local.producer.constraint_splits();
      for (std::size_t c = 0; c < workspace.cons_group.size(); ++c) {
        if (constraint_splits[c].size() == 0)
          continue;
        const auto group = workspace.cons_group[c];
        const auto ends = group_ends(job, group);
        const auto inverted =
            tf::arrangement::plane_recovery_flat_edge_inverted(ends);
        const std::array<Index, 2> feature{
            inverted ? ends[1] : ends[0], inverted ? ends[0] : ends[1]};
        const auto endpoints = group_endpoints(job, group);
        const auto source = std::uint8_t(source_of(job, group));
        for (const auto &choice : constraint_splits[c]) {
          const int shift =
              tf::exact::meta<Int>::param_bits - int(choice.depth);
          const param_t parameter = tf::exact::snap_parameter_to_edge<Int>(
              param_t(choice.numerator) << shift, endpoints[0], endpoints[1]);
          local.splits.push_back({feature, job.plane, group, parameter, source,
                                  std::uint8_t(inverted)});
        }
      }

      const auto begin = Index(local.steiner_points.size());
      const auto &frame = frame_of(job);
      local.producer.for_each_interior_steiner_point([&](const auto &point) {
        local.steiner_points.push_back(
            tf::exact::lift_plane_point(frame, point));
      });
      const auto end = Index(local.steiner_points.size());
      if (begin != end)
        local.steiner_runs.push_back({job.plane, begin, end});
    }
  };

  auto aggregate = [&](const local_t &local, const tf::none_t &) {
    tf::core::append(local.splits, plan.splits);
    tf::core::append(local.welds, plan.welds);
    tf::core::append(local.refused_planes, plan.refused_planes);
    const auto point_base = Index(steiner_points.size());
    for (const auto &run : local.steiner_runs)
      steiner_runs.push_back(
          {run[0], point_base + run[1], point_base + run[2], Index(0)});
    tf::core::append(local.steiner_points, steiner_points);
  };

  tf::blocked_reduce_sequenced_aggregate(jobs, tf::none, local_t{}, task,
                                         aggregate);

  plan.census.planes = jobs.size();
  plan.census.raw_boundary_proposals = plan.splits.size();

  if (plan.splits.size() != 0) {
    tbb::parallel_sort(
        plan.splits.begin(), plan.splits.end(),
        [](const tf::arrangement::plane_refinement_split<Index, Int> &a,
           const tf::arrangement::plane_refinement_split<Index, Int> &b) {
          if (a.feature != b.feature)
            return a.feature < b.feature;
          if (a.parameter != b.parameter)
            return a.parameter < b.parameter;
          const int a_source =
              a.source == tf::arrangement::plane_refinement_current_source ? 0
                                                                           : 1;
          const int b_source =
              b.source == tf::arrangement::plane_refinement_current_source ? 0
                                                                           : 1;
          if (a_source != b_source)
            return a_source < b_source;
          if (a.group != b.group)
            return a.group < b.group;
          if (a.plane != b.plane)
            return (a.plane) < (b.plane);
          return a.source < b.source;
        });

    const param_t maximum = param_t(1) << tf::exact::meta<Int>::param_bits;
    auto *write = plan.splits.begin();
    std::array<Index, 2> previous_feature{};
    param_t previous_parameter = param_t(0);
    bool has_previous = false;
    for (auto *read = plan.splits.begin(); read != plan.splits.end(); ++read) {
      if (!has_previous || read->feature != previous_feature) {
        previous_feature = read->feature;
        has_previous = false;
      }
      if (read->parameter <= param_t(0) || maximum <= read->parameter)
        continue;
      if (has_previous && read->parameter - previous_parameter <= param_t(1))
        continue;
      *write++ = *read;
      previous_parameter = read->parameter;
      has_previous = true;
    }
    plan.splits.erase_till_end(write);
  }
  plan.census.unique_boundary_splits = plan.splits.size();

  if (plan.welds.size() != 0) {
    tf::parallel_for_each(
        plan.welds,
        [](tf::arrangement::plane_flat_weld<Index> &weld) {
          if (weld.flat_1 < weld.flat_0)
            std::swap(weld.flat_0, weld.flat_1);
        },
        tf::checked);
    tbb::parallel_sort(plan.welds.begin(), plan.welds.end(),
                       [](const tf::arrangement::plane_flat_weld<Index> &a,
                          const tf::arrangement::plane_flat_weld<Index> &b) {
                         return std::tie(a.plane, a.flat_0, a.flat_1) <
                                std::tie(b.plane, b.flat_0, b.flat_1);
                       });
    plan.welds.erase_till_end(
        std::unique(plan.welds.begin(), plan.welds.end(),
                    [](const tf::arrangement::plane_flat_weld<Index> &a,
                       const tf::arrangement::plane_flat_weld<Index> &b) {
                      return a.plane == b.plane && a.flat_0 == b.flat_0 &&
                             a.flat_1 == b.flat_1;
                    }));
  }
  plan.census.projection_weld_relations = plan.welds.size();

  if (plan.refused_planes.size() != 0) {
    tbb::parallel_sort(plan.refused_planes.begin(), plan.refused_planes.end());
    plan.refused_planes.erase_till_end(
        std::unique(plan.refused_planes.begin(), plan.refused_planes.end()));
  }
  plan.census.refused_planes = plan.refused_planes.size();

  if (steiner_runs.size() != 0)
    tbb::parallel_sort(steiner_runs.begin(), steiner_runs.end());
  auto &steiner_offsets = plan.steiners.offsets_buffer();
  auto &owned_steiners = plan.steiners.data_buffer();
  steiner_offsets.allocate(std::size_t(plane_extent) + 1);
  std::fill(steiner_offsets.begin(), steiner_offsets.end(), Index(0));
  for (const auto &run : steiner_runs)
    steiner_offsets[std::size_t(run[0]) + 1] += run[2] - run[1];
  for (std::size_t plane = 1; plane < steiner_offsets.size(); ++plane)
    steiner_offsets[plane] += steiner_offsets[plane - 1];
  owned_steiners.allocate(std::size_t(steiner_offsets.back()));

  Index run_plane = Index(-1);
  Index run_output = Index(0);
  for (auto &run : steiner_runs) {
    if (run[0] != run_plane) {
      run_plane = run[0];
      run_output = steiner_offsets[std::size_t(run_plane)];
    }
    run[3] = run_output;
    run_output += run[2] - run[1];
  }
  tf::parallel_for_each(steiner_runs, [&](const auto &run) {
    std::copy(steiner_points.begin() + run[1], steiner_points.begin() + run[2],
              owned_steiners.begin() + run[3]);
  });
  plan.census.steiners = owned_steiners.size();

  return plan;
}

} // namespace tf::arrangement

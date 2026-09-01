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

#include "../../core/algorithm/deal_jobs.hpp"
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/vertex.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "../../topology/cdt_refine_config.hpp"
#include "./discover_plane_refinement.hpp"
#include "./plane_definition_source.hpp"
#include "./plane_recovery_birth.hpp"
#include "./plane_refinement_plan.hpp"
#include "./prepare_plane_triangulation.hpp"

#include <array>
#include <cstdint>

namespace tf::arrangement {

/// REFINEMENT. THE THIRD STATEMENT PRODUCER'S QUESTION: what would the refiner
/// change, asked of the arrangement as it stands, each plane read from the one
/// tier its ticket names. The discovery is a proposal, not a mutation —
/// its boundary splits enter the SAME wave every other statement enters,
/// and its Steiners are direct forms that never touch topology.
template <typename Index, typename Int, typename World, typename VertexOffsets,
          typename PointOfFlat, typename ExactPoint>
auto make_plane_refinement_plan(
    const World &world,
    const tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    const tf::buffer<Index> &plane_ticket, const VertexOffsets &vertex_offsets,
    Index n_flat, const tf::cdt_refine_config &config,
    const PointOfFlat &point_of_flat, const ExactPoint &exact_point)
    -> plane_refinement_plan<Index, Int> {
  using pt3_t = tf::exact::pt3<Int>;
  struct job_t {
    Index plane;
    std::uint8_t source;
  };
  tf::buffer<Index> order;
  tf::deal_jobs<Index>(
      tf::make_sequence_range(world.n_planes()),
      [&](Index plane) {
        return plane_definition_edge_count(world, local_tables, plane_ticket,
                                           plane);
      },
      order);
  tf::buffer<job_t> jobs;
  jobs.allocate(order.size());
  tf::parallel_transform(
      tf::make_range(order), jobs,
      [&](Index plane) {
        return job_t{plane,
                     find_plane_definition_source(world, local_tables,
                                                  plane_ticket, plane)
                             .immutable
                         ? plane_refinement_immutable_source
                         : plane_refinement_current_source};
      },
      tf::checked);
  const auto prepare_job = [&](const job_t &job, auto &workspace) {
    return prepare_plane_triangulation(world, local_tables, plane_ticket,
                                       vertex_offsets, job.plane, workspace,
                                       point_of_flat);
  };
  const auto definition = [&](const job_t &job, Index group) -> const auto & {
    return job.source == plane_refinement_current_source
               ? local_tables.canon_group(group)[0]
               : world.canon_group(group)[0];
  };
  const auto group_ends = [&](const job_t &job, Index group) {
    return plane_recovery_flat_edge_ends(definition(job, group), n_flat,
                                         vertex_offsets);
  };
  const auto group_endpoints = [&](const job_t &job, Index group) {
    const auto &def = definition(job, group);
    return std::array<pt3_t, 2>{exact_point(def.point_tag_0, def.point_0),
                                exact_point(def.point_tag_1, def.point_1)};
  };
  const auto frame_of = [&](const job_t &job) -> decltype(auto) {
    return world.frame(job.plane);
  };
  const auto source_of = [](const job_t &job, Index) { return job.source; };
  return discover_plane_refinement<Index, Int>(
      tf::make_range(jobs), world.n_planes(), n_flat, prepare_job, group_ends,
      group_endpoints, frame_of, source_of, config);
}

} // namespace tf::arrangement

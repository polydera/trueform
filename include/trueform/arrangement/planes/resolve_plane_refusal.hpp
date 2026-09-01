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

#include "../../core/buffer.hpp"
#include "../../exact/meta.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./collect_plane_corner_welds.hpp"
#include "./plane_definition_source.hpp"
#include "./prepare_plane_triangulation.hpp"
#include "./run_plane_cdt.hpp"
#include "./plane_tier_definitions.hpp"
#include "./state_plane_refusal.hpp"

namespace tf::arrangement {

/// WAVE. Triangulate a refusing plane again in resolve mode and let it state
/// what it saw. The tier the plane reads is the whole difference between a
/// statement that carries a root and one that cannot.
template <typename Index, typename Int, typename World, typename VertexOffsets,
          typename Local, typename PointOfFlat>
auto resolve_plane_refusal(
    const World &world,
    const tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    const tf::buffer<Index> &plane_ticket, const VertexOffsets &vertex_offsets,
    Index n_flat, Index crossing_kind, Index plane, Local &local,
    const PointOfFlat &point_of_flat) -> void {
  using param_t = typename tf::exact::meta<Int>::param_type;
  if (!prepare_plane_triangulation(world, local_tables, plane_ticket,
                                   vertex_offsets, plane, local, point_of_flat))
    return;
  if (!run_plane_cdt(local, world.member_count(plane) > Index(1), true))
    return;
  if (!collect_plane_corner_welds(local, local.cdt, plane, n_flat))
    return;
  state_plane_refusal(make_plane_tier_groups(world.tables(), local_tables),
                      plane, local.cdt,
                      local.ends, local.cons_group, n_flat, vertex_offsets,
                      crossing_kind,
                      param_t(1) << tf::exact::meta<Int>::param_bits,
                      local.statements, local.topology, local.census.landings,
                      local.census.crossings);
  local.census.collisions += local.cdt.constraint_collisions().size();
}

} // namespace tf::arrangement

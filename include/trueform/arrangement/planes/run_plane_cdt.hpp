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

#include "../../core/edges.hpp"
#include "../../core/range.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../topology/cdt/make_constrained_delaunay_full_span_aliases.hpp"
#include "../../topology/cdt_region_mode.hpp"
#include "./union_plane_constraint_aliases.hpp"

namespace tf::arrangement {

/// CORE. The stock kernel: one build per plane over the prepared constraint
/// set, its coincident constraints repaired after the fact — a collision the
/// aliases promote to a boundary re-enters the regions this same build holds.
template <typename Local>
auto run_plane_cdt(Local &local, bool pooled, bool resolve) -> bool {
  local.cdt.always_track_constraint_owners();
  const auto constraints =
      tf::make_edges(tf::make_blocked_range<2>(tf::make_range(local.cons)));
  const auto mode =
      pooled ? tf::cdt_region_mode::components : tf::cdt_region_mode::nesting;
  if (!local.cdt.build_triangulation(local.pts2.points(), constraints,
                                     tf::make_range(local.bnd), resolve, mode))
    return false;
  if (local.cdt.constraint_collisions().size() != 0) {
    tf::topology::cdt::make_constrained_delaunay_full_span_aliases(
        constraints, local.cdt.index_map().f(), local.cons_aliases,
        local.cons_alias_blocks);
    union_plane_constraint_aliases(local.cons_aliases, local.cons_alias_blocks,
                                   local.bnd, local.cons_row,
                                   local.cons_statements, local.folded_members,
                                   local.cons_boundary_promotions);
    if (local.cons_boundary_promotions.size() != 0)
      local.cdt.promote_constraint_boundaries(local.cons_boundary_promotions);
  }
  local.cdt.build_regions(mode);
  return true;
}

} // namespace tf::arrangement

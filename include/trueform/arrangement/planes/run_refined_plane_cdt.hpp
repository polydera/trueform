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
#include "../../core/offset_block_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../exact/vertex.hpp"
#include "../../topology/cdt/make_constrained_delaunay_full_span_aliases.hpp"
#include "../../topology/cdt_refine_config.hpp"
#include "../../topology/cdt_region_mode.hpp"
#include "./union_plane_constraint_aliases.hpp"
#include "./union_plane_constraint_collisions.hpp"

#include <cstddef>

namespace tf::arrangement {

/// REFINEMENT. p1 placed these with split freedom, on facts this arrangement
/// already paid for. The emission refiner is frozen, so it consumes them as
/// plain sites — no identity, no ancestry — and tops up what freezing still
/// allows. A plane p1 never reached seeds nothing.
template <typename Index, typename Int, typename World, typename Local>
auto seed_plane_refinement_sites(
    const World &world,
    const tf::offset_block_buffer<Index, tf::exact::pt3<Int>> &steiner_sites,
    Index plane, Local &local) -> void {
  if (std::size_t(plane) >= steiner_sites.size())
    return;
  const auto sites = steiner_sites[std::size_t(plane)];
  if (sites.size() == 0)
    return;
  const auto &frame = world.frame(plane);
  local.pts2.reserve(local.pts2.points().size() + sites.size());
  for (const auto &site : sites)
    local.pts2.emplace_back(site[frame.ax0], site[frame.ax1]);
}

/// REFINEMENT. Quality-refine one plane on a frozen constraint world: a split
/// would name a point this plane cannot give a shared identity, so a refiner
/// that states one refuses instead.
template <typename Index, typename Local>
auto refine_plane_triangulation(Local &local, bool pooled,
                                const tf::cdt_refine_config &config) -> bool {
  local.refiner.always_track_constraint_owners();
  const auto constraints =
      tf::make_edges(tf::make_blocked_range<2>(tf::make_range(local.cons)));
  local.cons_root.clear();
  if (!local.refiner.build(local.pts2.points(), constraints,
                           tf::make_range(local.bnd), config,
                           pooled ? tf::cdt_region_mode::components
                                  : tf::cdt_region_mode::nesting))
    return false;
  // a frozen constraint world states no split; a split would name a point
  // this plane cannot give a shared identity
  if (local.refiner.n_constraint_splits() != Index(0))
    return false;
  if (local.refiner.constraint_collisions().size() == 0)
    return true;
  tf::topology::cdt::make_constrained_delaunay_full_span_aliases(
      constraints, local.refiner.index_map().f(), local.cons_aliases,
      local.cons_alias_blocks);
  union_plane_constraint_aliases(local.cons_aliases, local.cons_alias_blocks,
                                 local.bnd, local.cons_row,
                                 local.cons_statements, local.folded_members,
                                 local.cons_boundary_promotions);
  if (local.cons_boundary_promotions.size() != 0)
    return false;
  union_plane_constraint_collisions(local.refiner.constraint_collisions(),
                                    Index(local.bnd.size()), local.cons_root);
  return true;
}

/// REFINEMENT. The refined kernel: ONE build per plane on the final constraint
/// world, emitted directly. Its regions are the seed's, so a plane whose
/// coincident aliases the stock kernel repairs after the fact cannot be
/// refined at all — it falls back, unrefined but exact.
template <typename Index, typename Int, typename World, typename Local>
auto run_refined_plane_cdt(
    const World &world,
    const tf::offset_block_buffer<Index, tf::exact::pt3<Int>> &steiner_sites,
    Index plane, Local &local, bool pooled,
    const tf::cdt_refine_config &config) -> bool {
  const auto n_prepared = local.pts2.points().size();
  seed_plane_refinement_sites(world, steiner_sites, plane, local);
  if (!refine_plane_triangulation<Index>(local, pooled, config)) {
    local.pts2.reallocate(n_prepared);
    return false;
  }
  return true;
}

} // namespace tf::arrangement

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

#include "../../topology/cdt_refiner.hpp"
#include "./find_plane_rim_landings.hpp"
#include "./flood_plane_member_coverage.hpp"
#include "./plane_orientation.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::arrangement {

/// What the acceptance of one triangulation states: whether it is accepted,
/// the winding it emits in its own projection, a pooled carrier's region
/// count, and the cuts the rim law could not land.
template <typename Index> struct plane_acceptance {
  bool accepted = false;
  int cdt_orientation = 1;
  Index n_regions = 0;
  std::size_t uncovered_cuts = 0;
};

/// THE ADMISSION LAW: whether a built triangulation is accepted for its
/// carrier is one predicate, asked by the round before it emits and by
/// refinement's discovery, of the seed its plan grows from, before it plans.
/// A plan therefore stands only on a seed the round would accept; whether
/// the build itself refused is its caller's.
///
/// The predicate reads the corners the caller's weld collection named
/// (`local.rep`): every corner of a triangulation that is not a refiner is a
/// prepared identity; a pooled carrier's regions all reach the exterior,
/// and the coverage that flood states is left in `local` for the emission;
/// and the rim law lands nothing (@ref find_plane_rim_landings), its
/// landings left in `local.rim_landings` as the refusal's statement.
template <typename Index, typename Int, typename Local, typename Mesh,
          typename Faces>
auto accept_plane_triangulation(Local &local, const Mesh &mesh,
                                const Faces &faces, bool pooled)
    -> plane_acceptance<Index> {
  plane_acceptance<Index> out;
  if constexpr (!std::is_same_v<std::decay_t<Mesh>,
                                tf::cdt_refiner<Index, Int, Int>>) {
    for (std::size_t v = 0; v < std::size_t(mesh.points().size()); ++v)
      if (local.rep[v] == Index(-1))
        return out;
  }
  out.cdt_orientation = plane_cdt_orientation<Int>(faces, mesh.points());
  const auto labels = mesh.region_labels();
  if (!pooled) {
    out.uncovered_cuts = find_plane_rim_landings<Index, Int>(
        mesh, local,
        [&labels](Index, Index triangle) {
          return labels[std::size_t(triangle)] != 0;
        },
        [&local](Index, Index constraint) {
          return (local.cons_winding[std::size_t(constraint)] & 1) != 0;
        },
        local.rim_walk, local.rim_landings);
    out.accepted = local.rim_landings.size() == 0;
    return out;
  }
  const auto n_regions = flood_plane_member_coverage<Index>(
      mesh, out.cdt_orientation, local.cons_row, local.cons_statements,
      Index(local.bnd.size()), local.region_adj, local.region_offsets,
      local.region_stack, local.region_reached, local.cover_range, local.cover,
      local.cover_winding);
  if (n_regions == Index(-1))
    return out;
  out.n_regions = n_regions;
  out.uncovered_cuts = find_plane_rim_landings<Index, Int>(
      mesh, local,
      [&](Index member, Index triangle) {
        const auto covering =
            local.cover_range[std::size_t(labels[std::size_t(triangle)])];
        for (auto at = covering[0]; at != covering[1]; ++at)
          if (local.cover[std::size_t(at)] == member)
            return true;
        return false;
      },
      [&local](Index member, Index constraint) {
        const auto row = local.cons_row[std::size_t(constraint)];
        for (auto at = row[0]; at != row[1]; ++at)
          if (local.cons_statements[std::size_t(at)].member == member)
            return (local.cons_statements[std::size_t(at)].winding & 1) != 0;
        return false;
      },
      local.rim_walk, local.rim_landings);
  out.accepted = local.rim_landings.size() == 0;
  return out;
}

} // namespace tf::arrangement

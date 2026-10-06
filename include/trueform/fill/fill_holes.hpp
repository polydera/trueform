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
#include "../core/coordinate_type.hpp"
#include "../core/none.hpp"
#include "../core/polygons.hpp"
#include "../exact/pt_converter.hpp"
#include "../exact/resolve_int_type.hpp"
#include "../topology/boundary_rims.hpp"
#include "../topology/face_membership.hpp"
#include "../topology/policy/face_membership.hpp"
#include "./holes/fill_hole_rims.hpp"
#include "./hole_fill_config.hpp"
#include "./hole_fill_result.hpp"
#include <cstddef>

namespace tf {

/// @ingroup fill
/// @brief Fill a mesh's boundary rims with patches of its own kind.
///
/// One rim is one group, and a group is filled on its own: its cycle is
/// validated and rooted, then handed down a ladder of tiers — exhaustive
/// enumeration for a rim of at most five vertices, the constrained Delaunay
/// refiner in the rim's own plane where it has one, and otherwise the
/// Delaunay tetrahedralization of its sites, over which a two-stage table
/// states the patch of least bottleneck angle and, under that, least area.
/// Rims run in parallel against each other, and inside one rim its facet
/// statement and the table's band stages fan out again; a small rim stays
/// serial under their own cutoffs.
///
/// Every tier works on the exact lattice the input converts to, on a scale
/// that makes each dyadic split of an original edge a lattice point of its
/// own. A tier that needs such a split publishes it into one canonical split
/// table, and the carrying triangles those splits reach are published beside
/// it as face plans — the one authority on what those faces become, which
/// @ref tf::make_filled_mesh replays.
///
/// A carrying face must be a TRIANGLE; a rim edge carried by anything else
/// refuses its group. The fill works in the form's own coordinates and a
/// tagged frame does not enter it, so the minted points stand in the same
/// space as the input's own.
///
/// @tparam Int The lattice the exact predicates dispatch on; defaults per
///   the input's coordinate type through @ref tf::exact::resolve_int_type.
/// @tparam Policy The polygons policy type.
/// @tparam Index The index type of the rims and the mesh.
/// @param polygons The mesh the rims are read against.
/// @param rims The rims to fill, the @ref tf::make_boundary_rims shape.
/// @param config How the patches are shaped.
/// @return The @ref tf::hole_fill_result, one group per rim.
template <typename Int = tf::none_t, typename Policy, typename Index>
auto fill_holes(const tf::polygons<Policy> &polygons,
                const tf::boundary_rims<Index> &rims,
                const tf::hole_fill_config &config = {})
    -> tf::hole_fill_result<Index, tf::coordinate_type<Policy>> {
  using RealT = tf::coordinate_type<Policy>;
  using Lattice = tf::exact::resolve_int_type<Int, RealT>;

  tf::hole_fill_result<Index, RealT> result;
  result.group_offsets.allocate(rims.size() + 1);
  for (std::size_t group = 0; group <= rims.size(); ++group)
    result.group_offsets[group] = Index(group);
  if (!rims.size())
    return result;

  const auto converter =
      tf::exact::make_pt_converter<Lattice, RealT>(polygons.points());
  if constexpr (tf::has_face_membership_policy<Policy>) {
    tf::fill::fill_hole_rims<Lattice>(polygons, polygons.face_membership(),
                                      rims, converter, config, result);
  } else {
    tf::face_membership<Index> membership;
    membership.build(polygons);
    tf::fill::fill_hole_rims<Lattice>(polygons, membership, rims, converter,
                                      config, result);
  }
  return result;
}

} // namespace tf

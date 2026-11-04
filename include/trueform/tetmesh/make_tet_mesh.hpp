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
#include "../core/none.hpp"
#include "../core/polygons.hpp"
#include "../core/resolved_output_real.hpp"
#include "./entrance/make_tet_entrance.hpp"
#include "./entrance/measure_tet_corner_need.hpp"
#include "./entrance/tet_entrance_rung_types.hpp"
#include "./mesh/mesh_tet_entrance.hpp"
#include "./tet_mesh_report.hpp"
#include <type_traits>

namespace tf {

/// @ingroup tetmesh
/// @brief Tetrahedralize the solid a closed surface winds.
///
/// Crossing faces are cut at their canonical intersections, every face and
/// edge is recovered exactly as a constraint, and the complex is carved by
/// the surface's winding, the cells it winds positively kept. The input
/// converts once onto the house converter's lattice for the cheapest
/// instantiation that holds its precision, less the ten bits its frames
/// take: int32 sites stored in int64 for a float, int64 for a double. A
/// constraint the frame splits too coarsely is recovered below it, a site
/// standing as deep as the instantiation's deep lattice admits, and every
/// emitted coordinate is a site's position mapped back. The sites a run may
/// mint are bounded, so an input whose recovery would not end refuses for
/// the budget. Coplanar input triangles that meet wound alike are one facet,
/// recovered as one region whose interior the mesh triangulates freely.
/// Returns the tetrahedra; the output triangles covering each facet, block
/// `f` for facet `f`, wound as it is; the facets each input face lies in,
/// block `f` for input face `f`, ascending, one unless a crossing cut it;
/// and the report, which states the lattice that ran and the sites below
/// its frame. `Int`, when stated, restricts the instantiation to that
/// lattice; `OutputCoordinateType` is the coordinate the mesh is emitted in.
template <typename Int = tf::none_t, typename OutputCoordinateType = tf::none_t,
          typename Policy>
auto make_tet_mesh(const tf::polygons<Policy> &polygons) {
  using RealT = tf::coordinate_type<Policy>;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, RealT>;
  static_assert(std::is_floating_point_v<RealOut>,
                "a tet mesh is emitted in floating-point coordinates");
  constexpr bool home = tf::tetmesh::tet_entrance_rung_admits<0, Int, RealT>;
  static_assert(home || tf::tetmesh::tet_entrance_rung_admits<1, Int, RealT>,
                "no rung's lattice holds the input's precision");
  using Index = std::decay_t<decltype(polygons.faces()[0][0])>;
  using types = tf::tetmesh::tet_entrance_rung_types<home ? 0 : 1>;
  const auto need = tf::tetmesh::measure_tet_corner_need(polygons);
  tf::tet_mesh_report report;
  report.corner_bits_needed = need.corner_bits;
  report.min_angle = need.min_angle;
  report.rung = home ? 0 : 1;
  const auto entrance =
      tf::tetmesh::make_tet_entrance<Index, typename types::int_type,
                                     typename types::coord_type>(polygons);
  return tf::tetmesh::mesh_tet_entrance<RealOut>(polygons, entrance, report);
}

} // namespace tf

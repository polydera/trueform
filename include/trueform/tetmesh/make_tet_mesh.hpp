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
#include "../core/return_report.hpp"
#include "../reindex/return_index_map.hpp"
#include "./mesh/mesh_tet_surface.hpp"
#include "./tet_mesh_config.hpp"
#include "./tet_mesh_index_map.hpp"

namespace tf {

/// @ingroup tetmesh
/// @brief Tetrahedralize the solid a closed triangle surface winds.
///
/// The input is a triangle mesh. Every face and edge is recovered exactly as
/// a constraint, and the complex's regions are labeled by the surface's
/// winding, the regions it winds positively kept. A crossing is found where
/// its recovery refuses it, and the faces it reaches are cut there, at their
/// canonical intersections, and recovered again on the standing complex,
/// until no crossing is named or a round's point is one the input cannot tell
/// from a point an earlier round minted, where the run refuses; the
/// call is fastest on a surface that crosses little, and one known to cross
/// much is best arranged first (@ref tf::make_polygon_arrangements with
/// integral output). Coincident points weld and a vertex on another face's
/// edge splits it, so a seam the input leaves open but its positions close
/// meshes. The input converts once onto the house converter's
/// lattice, fitted to the points its faces name, for the cheapest instantiation
/// that holds its precision, less the ten bits its frames take: int32 sites
/// stored in int64 for a float, int64 for a double. A constraint the frame
/// splits too coarsely is recovered below it, a site standing as deep as the
/// instantiation's deep lattice admits, and every emitted coordinate is a
/// site's position mapped back. The sites a run may mint are bounded, so an
/// input whose recovery would not end refuses for the budget. Coplanar input
/// triangles that meet wound alike are one facet, recovered as one region whose
/// interior the mesh triangulates freely. Returns the tetrahedra and the
/// @ref tf::tet_mesh_status; under @ref tf::return_report the
/// @ref tf::tet_mesh_report in its place, which names a refusal's reason,
/// states the lattice that ran and the sites below its frame, and measures
/// the diagnostics no status needs.
/// `Int`, when stated, restricts the instantiation to that lattice;
/// `OutputCoordinateType` is the coordinate the mesh is emitted in, double
/// unless stated: the points are lattice positions mapped back, 40 bits for a
/// float input, and a float export reads some sliver cells inverted (@ref
/// tf::tet_mesh_report::inverted). `config.split_constraints` resolves the
/// surface, and `false` takes it as given: no drain cuts (`drains` is zero),
/// and a crossing meets the recovery as a constraint it cannot recover, so
/// the run refuses there. `config.tolerance`, in the input's coordinates, is
/// plane pooling: the entrance pools near-coplanar faces onto one plane, and
/// it is the band every drain states its contacts in. A
/// tagged frame places the points; a reflecting one flips the winding the
/// region labeling reads, so the call returns `meshed` with no cell and
/// `negative_cells` counts what it would have kept: winding is semantic, and
/// the caller re-winds. A tagged link and frame are read, a tagged tree and
/// membership are not.
template <typename Int = tf::none_t, typename OutputCoordinateType = double,
          typename Policy>
auto make_tet_mesh(const tf::polygons<Policy> &polygons,
                   const tf::tet_mesh_config &config = {}) {
  return tf::tetmesh::mesh_tet_surface<Int, OutputCoordinateType>(
      polygons, config, tf::none, tf::none);
}

/// @ingroup tetmesh
/// @brief Tetrahedralize the solid a closed surface winds, with the
///        @ref tf::tet_mesh_index_map of the input surface onto its boundary.
template <typename Int = tf::none_t, typename OutputCoordinateType = double,
          typename Policy>
auto make_tet_mesh(const tf::polygons<Policy> &polygons,
                   const tf::tet_mesh_config &config,
                   tf::return_index_map_t tag) {
  return tf::tetmesh::mesh_tet_surface<Int, OutputCoordinateType>(
      polygons, config, tf::none, tag);
}

/// @ingroup tetmesh
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = double,
          typename Policy>
auto make_tet_mesh(const tf::polygons<Policy> &polygons,
                   tf::return_index_map_t tag) {
  return make_tet_mesh<Int, OutputCoordinateType>(polygons,
                                                  tf::tet_mesh_config{}, tag);
}

/// @ingroup tetmesh
/// @brief Tetrahedralize the solid a closed surface winds, with the
///        @ref tf::tet_mesh_report in place of the status.
template <typename Int = tf::none_t, typename OutputCoordinateType = double,
          typename Policy>
auto make_tet_mesh(const tf::polygons<Policy> &polygons,
                   const tf::tet_mesh_config &config,
                   tf::return_report_t report) {
  return tf::tetmesh::mesh_tet_surface<Int, OutputCoordinateType>(
      polygons, config, report, tf::none);
}

/// @ingroup tetmesh
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = double,
          typename Policy>
auto make_tet_mesh(const tf::polygons<Policy> &polygons,
                   tf::return_report_t report) {
  return make_tet_mesh<Int, OutputCoordinateType>(
      polygons, tf::tet_mesh_config{}, report);
}

/// @ingroup tetmesh
/// @brief Tetrahedralize the solid a closed surface winds, with the
///        @ref tf::tet_mesh_report and the @ref tf::tet_mesh_index_map.
template <typename Int = tf::none_t, typename OutputCoordinateType = double,
          typename Policy>
auto make_tet_mesh(const tf::polygons<Policy> &polygons,
                   const tf::tet_mesh_config &config,
                   tf::return_report_t report, tf::return_index_map_t tag) {
  return tf::tetmesh::mesh_tet_surface<Int, OutputCoordinateType>(
      polygons, config, report, tag);
}

/// @ingroup tetmesh
/// @overload
template <typename Int = tf::none_t, typename OutputCoordinateType = double,
          typename Policy>
auto make_tet_mesh(const tf::polygons<Policy> &polygons,
                   tf::return_report_t report, tf::return_index_map_t tag) {
  return make_tet_mesh<Int, OutputCoordinateType>(
      polygons, tf::tet_mesh_config{}, report, tag);
}

} // namespace tf

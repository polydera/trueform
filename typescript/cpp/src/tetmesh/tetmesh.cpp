/*
 * Copyright (c) 2025 XLAB
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

#include "./tetmesh_impl.hpp"
#include <emscripten/bind.h>

// The result types carry no input dtype, so they register once.
EMSCRIPTEN_BINDINGS(trueform_tetmesh) {
  using namespace tf::ts;

  emscripten::value_object<tet_face_census_t>("TetFaceCensus")
      .field("covered", &tet_face_census_t::covered)
      .field("passes", &tet_face_census_t::passes)
      .field("flipped", &tet_face_census_t::flipped)
      .field("points", &tet_face_census_t::points)
      .field("unplaced", &tet_face_census_t::unplaced)
      .field("onPiece", &tet_face_census_t::on_piece)
      .field("refused", &tet_face_census_t::refused)
      .field("openRim", &tet_face_census_t::open_rim)
      .field("crossed", &tet_face_census_t::crossed)
      .field("offLattice", &tet_face_census_t::off_lattice)
      .field("budget", &tet_face_census_t::budget)
      .field("pierced", &tet_face_census_t::pierced)
      .field("standing", &tet_face_census_t::standing);

  emscripten::value_object<tet_constraint_census_t>("TetConstraintCensus")
      .field("segmentRounds", &tet_constraint_census_t::segment_rounds)
      .field("shellSplits", &tet_constraint_census_t::shell_splits)
      .field("segmentsRefused", &tet_constraint_census_t::segments_refused)
      .field("unrestoredFacets", &tet_constraint_census_t::unrestored_facets)
      .field("faces", &tet_constraint_census_t::faces)
      .field("flipRecovered", &tet_constraint_census_t::flip_recovered)
      .field("searchRecovered", &tet_constraint_census_t::search_recovered)
      .field("splitRows", &tet_constraint_census_t::split_rows)
      .field("hullBarred", &tet_constraint_census_t::hull_barred)
      .field("restoredBeforeFaces",
             &tet_constraint_census_t::restored_before_faces)
      .field("retiredCrossing", &tet_constraint_census_t::retired_crossing)
      .field("retiredNeighbour", &tet_constraint_census_t::retired_neighbour)
      .field("revivedSegments", &tet_constraint_census_t::revived_segments)
      .field("collapsed", &tet_constraint_census_t::collapsed)
      .field("collapseRefused", &tet_constraint_census_t::collapse_refused)
      .field("stalledDrains", &tet_constraint_census_t::stalled_drains)
      .field("budgetDrains", &tet_constraint_census_t::budget_drains);

  emscripten::value_object<tet_mesh_report_t>("TetMeshReport")
      .field("status", &tet_mesh_report_t::status)
      .field("refusedSegments", &tet_mesh_report_t::refused_segments)
      .field("refusedDepth", &tet_mesh_report_t::refused_depth)
      .field("refusedBudget", &tet_mesh_report_t::refused_budget)
      .field("refusedFaces", &tet_mesh_report_t::refused_faces)
      .field("rung", &tet_mesh_report_t::rung)
      .field("drains", &tet_mesh_report_t::drains)
      .field("pairsNamed", &tet_mesh_report_t::pairs_named)
      .field("pairsCut", &tet_mesh_report_t::pairs_cut)
      .field("trianglesCut", &tet_mesh_report_t::triangles_cut)
      .field("droppedDegenerate", &tet_mesh_report_t::dropped_degenerate)
      .field("stitchedTriangles", &tet_mesh_report_t::stitched_triangles)
      .field("failedTriangles", &tet_mesh_report_t::failed_triangles)
      .field("pairsUnresolved", &tet_mesh_report_t::pairs_unresolved)
      .field("inputOpenEdges", &tet_mesh_report_t::input_open_edges)
      .field("recoveredOpenEdges", &tet_mesh_report_t::recovered_open_edges)
      .field("storageBits", &tet_mesh_report_t::storage_bits)
      .field("inputBits", &tet_mesh_report_t::input_bits)
      .field("frameBits", &tet_mesh_report_t::frame_bits)
      .field("deepSites", &tet_mesh_report_t::deep_sites)
      .field("deepest", &tet_mesh_report_t::deepest)
      .field("cornerBitsNeeded", &tet_mesh_report_t::corner_bits_needed)
      .field("minAngle", &tet_mesh_report_t::min_angle)
      .field("flat", &tet_mesh_report_t::flat)
      .field("inverted", &tet_mesh_report_t::inverted)
      .field("negativeCells", &tet_mesh_report_t::negative_cells)
      .field("census", &tet_mesh_report_t::census);

  emscripten::value_object<tet_mesh_t>("TetMesh")
      .field("tetrahedra", &tet_mesh_t::tetrahedra)
      .field("points", &tet_mesh_t::points)
      .field("status", &tet_mesh_t::status)
      .field("report", &tet_mesh_t::report);

  emscripten::value_object<tet_mesh_with_index_map_t>("TetMeshWithIndexMap")
      .field("tetrahedra", &tet_mesh_with_index_map_t::tetrahedra)
      .field("points", &tet_mesh_with_index_map_t::points)
      .field("status", &tet_mesh_with_index_map_t::status)
      .field("report", &tet_mesh_with_index_map_t::report)
      .field("facetOffsets", &tet_mesh_with_index_map_t::facet_offsets)
      .field("facetTriangles", &tet_mesh_with_index_map_t::facet_triangles)
      .field("faceFacetOffsets", &tet_mesh_with_index_map_t::face_facet_offsets)
      .field("faceFacets", &tet_mesh_with_index_map_t::face_facets);
}

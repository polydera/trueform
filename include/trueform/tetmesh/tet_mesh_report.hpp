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
#include "../topology/cdt/dt3/recovery/tet_constraint_census.hpp"
#include "./tet_mesh_status.hpp"
#include <cstddef>

namespace tf {

/// The status of a @ref tf::make_tet_mesh call, its refusals by kind, the
/// constraint recovery's census, and the lattice the mesh ran on.
///
/// `rung` is the instantiation that ran, its sites `storage_bits` wide, the
/// lattice `input_bits` and the frame `frame_bits`; `deep_sites` counts the
/// sites the recovery placed below the frame, the deepest of them
/// `deepest` bits below it. `corner_bits_needed` and `min_angle` are the
/// frame depth the input's narrowest corner proposes and its angle in
/// degrees, reported, never chosen by. `refused_depth` counts the
/// constraints whose frames ran out at the deep lattice's cap: the segments
/// with no split left, and the faces whose lattice names no new point, their
/// point standing already or none lying in the face. `refused_budget`
/// counts the segments whose splits would have passed the site budget.
/// `entrance_passes` counts the arrangement passes the entrance took,
/// `dropped_degenerate` the triangles it threw out as an edge or a line, and
/// `stitched_triangles` those it added subdividing their neighbours.
/// `entrance_failed_planes` and `entrance_failed_faces` count the carriers
/// an entrance pass left without triangles and the faces bounding area on
/// them; the run refuses there, before any site is placed.
/// `entrance_open_edges` counts the edges the entrance's triangles leave
/// open, and an `open` or `unwound` run counts the input's in
/// `input_open_edges`, so it states whether the input came open or the
/// entrance opened it.
///
/// Every kept cell is non-negative on the lattice: `flat` counts those whose
/// sites are coplanar there, and `inverted` those an export narrower than
/// the lattice reads negative. `negative_cells` counts the cells the carve
/// winds negatively, which a surface wound inward encloses in place of a
/// mesh.
struct tet_mesh_report {
  tet_mesh_status status = tet_mesh_status::meshed;
  std::size_t refused_segments = 0;
  std::size_t refused_depth = 0;
  std::size_t refused_budget = 0;
  std::size_t refused_faces = 0;
  std::size_t rung = 0;
  std::size_t entrance_passes = 0;
  std::size_t dropped_degenerate = 0;
  std::size_t stitched_triangles = 0;
  std::size_t entrance_failed_planes = 0;
  std::size_t entrance_failed_faces = 0;
  std::size_t input_open_edges = 0;
  std::size_t entrance_open_edges = 0;
  int storage_bits = 0;
  int input_bits = 0;
  int frame_bits = 0;
  std::size_t deep_sites = 0;
  int deepest = 0;
  int corner_bits_needed = 0;
  double min_angle = 180;
  std::size_t flat = 0;
  std::size_t inverted = 0;
  std::size_t negative_cells = 0;
  tf::topology::cdt::dt3::tet_constraint_census census;
};

} // namespace tf

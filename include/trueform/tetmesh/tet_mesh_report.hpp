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
/// counts the segments whose splits would have passed the site budget, the
/// one budget a drain's mints consume too.
/// `dropped_degenerate` counts the triangles the surface's statements threw
/// out as an edge or a line, and `stitched_triangles` those they added
/// subdividing their neighbours. A crossing is found where the recovery's
/// walk refuses it and is cut in a drain, on the standing complex: `drains`
/// counts the drains that cut, zero on a surface that crosses nowhere,
/// `pairs_named` the crossings the walks named, `pairs_cut` those a drain
/// cut, and `triangles_cut` the triangles the drains traced; the census
/// counts the facets the drains retired. `failed_triangles` counts the
/// carriers a drain's wave left refusing and the T-junctions it would leave;
/// the run refuses there. Every drain either mints, consuming the site
/// budget, or strictly reduces the crossings it was handed, so the drains
/// are finite; a drain that does neither refuses the run, counted in the
/// census as `stalled_drains`, and mints the budget cannot hold refuse it as
/// `budget_drains`. `pairs_unresolved` counts the crossings the last walk
/// named that no drain cut — the surface taken as given, a drain that
/// retired no facet, or a refused drain; the run refuses there. Where the
/// lattice cannot resolve what the drains keep naming, the caller states
/// the deeper one, `tf::make_tet_mesh<std::int64_t>`, which resolves them
/// where it carries the room the input's own precision leaves.
/// `recovered_open_edges` counts the edges the recovered surface leaves
/// open — read on the faces' recovered facets, after the build welded
/// coincident sites and the segments split at the sites standing on them,
/// wherever the entrance's triangles left any — and an `open` or `unwound`
/// run counts the input's in `input_open_edges`. Which of the two the run
/// refuses as is read off the rims those free edges form: a hole is `open`,
/// a flap's free rim `unwound`, the flap being a sheet that bounds nothing.
///
/// Every kept cell is non-negative on the lattice: `flat` counts those whose
/// sites are coplanar there, and `inverted` those an export narrower than
/// the lattice reads negative. `negative_cells` counts the cells the region
/// labeling winds negatively, which a surface wound inward encloses in place of
/// a mesh.
struct tet_mesh_report {
  tet_mesh_status status = tet_mesh_status::meshed;
  std::size_t refused_segments = 0;
  std::size_t refused_depth = 0;
  std::size_t refused_budget = 0;
  std::size_t refused_faces = 0;
  std::size_t rung = 0;
  std::size_t drains = 0;
  std::size_t pairs_named = 0;
  std::size_t pairs_cut = 0;
  std::size_t triangles_cut = 0;
  std::size_t dropped_degenerate = 0;
  std::size_t stitched_triangles = 0;
  std::size_t failed_triangles = 0;
  std::size_t pairs_unresolved = 0;
  std::size_t input_open_edges = 0;
  std::size_t recovered_open_edges = 0;
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

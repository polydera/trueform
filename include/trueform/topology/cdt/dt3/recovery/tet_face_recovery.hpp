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
#include "../../../../core/buffer.hpp"
#include "../../../../core/offset_block_buffer.hpp"
#include "../tet_execution_tuning.hpp"
#include "./regions/tet_round_workspace.hpp"
#include "./tet_face_census.hpp"
#include "./tet_face_cover_scratch.hpp"
#include "./tet_face_crossing.hpp"
#include "./tet_face_status.hpp"
#include "./tet_rim_piece.hpp"
#include "./tet_site_cell_scratch.hpp"
#include "./tet_stamps.hpp"
#include <array>

namespace tf::topology::cdt::dt3 {

/// Face recovery's product and the scratch its passes reuse. A face is one
/// planar polygon: its plane its first triangle's, its rim the pieces the
/// segment phase recovered for its segments, every one standing exactly on
/// its segment and so on the plane; its interior triangulation is nobody's
/// constraint. A face is recovered when the facets of the complex on its
/// plane inside its rim tile it, which they do exactly when its cover leaves
/// no frontier, and is permanent through one
/// invariant: every facet of its cover stands in the lane owned by the face,
/// and a protected facet is never flipped. `status` per face; `cover` the
/// tickets of each face's facets found before any flip, a block per face,
/// `cover_layers` aligned with them the layers of the face over each
/// (@ref state_tet_cover_layers), and `covered` whether they tile it;
/// `chain_passes` is the passes one face's chain may take before it refuses
/// as `capped` (@ref tet_execution_tuning::recovery_face_passes).
/// `n_regions` is the regions the partition named, `round_work` the
/// workspace one round of them walks on and `region_left` the positions the
/// admission stopped (@ref run_tet_region_round); `deferred` the
/// faces one barred, which the frontier walks on the thawed complex;
/// `layers` the serial pass's own; `uncovered` the faces the serial pass
/// takes; `site_cells` the cells of every site of `star_sites`, the corners
/// of the faces whose facet is found through them (@ref state_tet_site_cells),
/// stated on `site_scratch`; `cover_scratch` the serial pass's
/// lists for the cover it reads (@ref tet_face_cover_scratch), with
/// `touched` and `seeds` its advance's own
/// (@ref advance_tet_face_cover), and
/// `standing` the piercing edges whose point a site held in the pass, each
/// with a cell near it, and `piercing` every piercing edge the pass met;
/// `exhausted` the standing edges of the face the serial pass is on whose
/// search below the frame found no point at any depth, none of the rows
/// around them written since; `crossing` per face the constraint a refused
/// face crosses (@ref tet_face_crossing), and `deep` whether its refusal is
/// the deep lattice's: no point to the cap, or every point standing; both
/// are stated afresh each phase a face is due, a face not due keeping the
/// last phase's, and only a refused face's mean anything.
/// `near_stamps` and `reached` are the cells a point's reach floods
/// (@ref find_tet_site_within).
template <typename Index, typename Coord, typename Int> struct tet_face_recovery {
  std::size_t chain_passes = tet_execution_tuning::recovery_face_passes;
  tf::buffer<tet_face_status> status;
  tf::offset_block_buffer<Index, Index> cover;
  tf::buffer<Index> cover_layers;
  tf::buffer<Index> hold_face;
  tf::buffer<Index> layers;
  tf::buffer<char> covered;
  std::size_t n_regions = 0;
  tet_round_workspace<Index> round_work;
  tf::buffer<Index> region_left;
  std::size_t region_tasks = 0;
  std::size_t region_deferred = 0;
  tf::buffer<Index> deferred;
  tf::buffer<Index> uncovered;
  tf::offset_block_buffer<Index, Index> site_cells;
  tf::buffer<Index> star_sites;
  tet_site_cell_scratch<Index> site_scratch;
  tet_face_cover_scratch<Index> cover_scratch;
  tf::buffer<std::array<Index, 2>> touched;
  tf::buffer<tet_rim_piece<Index>> seeds;
  tf::buffer<std::array<Index, 3>> standing;
  tf::buffer<std::array<Index, 2>> piercing;
  tf::buffer<std::array<Index, 2>> exhausted;
  tf::buffer<tet_face_crossing<Index>> crossing;
  tf::buffer<char> deep;
  tet_stamps near_stamps;
  tf::buffer<Index> reached;
  tet_face_census census;
};

} // namespace tf::topology::cdt::dt3

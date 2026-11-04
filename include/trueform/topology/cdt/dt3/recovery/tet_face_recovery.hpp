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
#include "./tet_face_census.hpp"
#include "./tet_face_status.hpp"
#include "./tet_rim_piece.hpp"
#include <array>

namespace tf::topology::cdt::dt3 {

/// Face recovery's product and the scratch its passes reuse. A face is one
/// planar polygon: its plane its first triangle's, its rim the pieces the
/// segment phase recovered for its segments, every one standing exactly on
/// its segment and so on the plane; its interior triangulation is nobody's
/// constraint. A face is recovered when the facets of the complex on its
/// plane inside its rim tile it, read by area, and is permanent through one
/// invariant: every facet of its cover stands in the lane owned by the face,
/// and a protected facet is never flipped. `status` per face; `cover` the
/// tickets of each face's facets found before any flip, a block per face,
/// and `covered` whether they tile it; `uncovered` the faces the serial pass
/// takes; `points` every face point placed as `(face, site)`, sorted at the
/// end; `rim`, `tickets` and `frontier` the serial pass's scratch, and
/// `standing` the piercing edges whose point a site held in the pass, each
/// with a cell near it.
template <typename Index, typename Coord, typename Int> struct tet_face_recovery {
  tf::buffer<tet_face_status> status;
  tf::offset_block_buffer<Index, Index> cover;
  tf::buffer<char> covered;
  tf::buffer<Index> uncovered;
  tf::buffer<std::array<Index, 2>> points;
  tf::buffer<tet_rim_piece<Index>> rim;
  tf::buffer<Index> tickets;
  tf::buffer<std::array<Index, 2>> frontier;
  tf::buffer<std::array<Index, 3>> standing;
  tet_face_census census;
};

} // namespace tf::topology::cdt::dt3

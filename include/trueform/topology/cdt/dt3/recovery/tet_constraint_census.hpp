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
#include "./tet_face_census.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// What constraint recovery did: the rounds phase S ran, the splits its
/// corner shells proposed and the segments it refused, the edges its
/// restorations left not locally Delaunay, every face route of phase F, and
/// the segment rows flips recovered without and with the full search, the
/// rows that split, and the rings a hull cell kept from flipping; and the
/// facets the drains retired, those holding a traced triangle and those a
/// split beyond reached (@ref tf::tetmesh::cut_tet_pairs), and the recovered
/// segments a drain's sites took an edge of, recovered again
/// (@ref revive_tet_segments); and the sites the census collapsed into an
/// input corner and the collapses it refused
/// (@ref tf::tetmesh::collapse_tet_steiner_corners), a census its round cap
/// cut short counting the faces its last round moved among the refused; and
/// the drains refused as stalled — one that neither minted nor reduced the
/// crossings it was handed — and for the site budget
/// (@ref tf::tetmesh::tet_drain_measure). Every count is cumulative over the
/// run, a constraint due again counted again; what stands at the end is the
/// statuses' (@ref tf::tet_mesh_report).
///
/// The restoration follows the region labeling and reads only the cells it
/// keeps, except where phase S damaged the complex densely enough to restore
/// its own rows first, which `restored_before_faces` states
/// (@ref recover_tet_segment_phase).
struct tet_constraint_census {
  std::size_t segment_rounds = 0;
  std::size_t shell_splits = 0;
  std::size_t segments_refused = 0;
  std::size_t segments_capped = 0;
  std::size_t region_tasks = 0;
  std::size_t region_deferred = 0;
  std::size_t unrestored_facets = 0;
  tet_face_census faces;
  std::size_t flip_recovered = 0;
  std::size_t search_recovered = 0;
  std::size_t split_rows = 0;
  std::size_t hull_barred = 0;
  bool restored_before_faces = false;
  std::size_t retired_crossing = 0;
  std::size_t retired_neighbour = 0;
  std::size_t revived_segments = 0;
  std::size_t collapsed = 0;
  std::size_t collapse_refused = 0;
  std::size_t stalled_drains = 0;
  std::size_t budget_drains = 0;
};

} // namespace tf::topology::cdt::dt3

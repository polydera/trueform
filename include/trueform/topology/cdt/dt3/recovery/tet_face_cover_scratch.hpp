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
#include "../../../../core/hash_map.hpp"
#include "../tet_edge_star.hpp"
#include "./tet_cover_due.hpp"
#include "./tet_rim_piece.hpp"
#include <array>

namespace tf::topology::cdt::dt3 {

/// The lists one read of a face's cover walks on, stated once for every
/// pass that reads one — the parallel first read's block state and the
/// serial pass's alike: `rim` the pieces the read spreads from, `tickets`
/// the facets it finds and `frontier` the rim it leaves
/// (@ref read_tet_face_cover); `check_tickets` and `check_frontier` the
/// second read the cover check makes (@ref check_tet_face_cover); `star`
/// the cells an edge is found through (@ref find_tet_edge); `seen` the
/// tickets a spread has taken, scanned while they are few and entered into
/// `hashed` once a cover outgrows the scan, each with its position, and
/// `due` the facets it owes their own edges, `adjacent` the facet across
/// each of their sides (@ref spread_tet_face_cover); and the tables the
/// layers of a cover are flooded on (@ref state_tet_cover_layers):
/// `rim_sides`, `steps`, `stack`, `winding`, with
/// `check_layers` the second read's.
template <typename Index> struct tet_face_cover_scratch {
  tf::buffer<tet_rim_piece<Index>> rim;
  tf::buffer<Index> tickets;
  tf::buffer<tet_rim_piece<Index>> frontier;
  tf::buffer<Index> check_tickets;
  tf::buffer<tet_rim_piece<Index>> check_frontier;
  tet_edge_star<Index> star;
  tf::buffer<Index> seen;
  tf::buffer<tet_cover_due<Index>> due;
  tf::hash_map<Index, Index> hashed;
  tf::buffer<std::array<Index, 3>> adjacent;
  tf::buffer<std::array<Index, 4>> rim_sides;
  tf::buffer<signed char> steps;
  tf::buffer<Index> stack;
  tf::buffer<signed char> winding;
  tf::buffer<Index> check_layers;
};

} // namespace tf::topology::cdt::dt3

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
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/vector.hpp"
#include "../../exact/vertex.hpp"
#include "../../topology/cdt/dt3/tet_edge_query_workspace.hpp"
#include "../../topology/cdt/dt3/tet_site_candidate.hpp"
#include "../../topology/delaunay_tetrahedralizer.hpp"
#include "./hole_lattice.hpp"
#include "./hole_rim.hpp"
#include <array>

namespace tf::fill {

/// The tables one rim's two-stage solve stands on. A facet of the complex
/// carries its own reading — the plane it states, the area it adds and the
/// host angles only it can be charged — and a STATE is a facet's chord read
/// from one side, which is why there are two per facet and one more for the
/// root.
///
/// THE CHORD IS THE JOIN: a facet `p q r` is a candidate of the chord
/// `p r` with `q` inside it, so the candidates group by that chord once and
/// every state reads its own group through `block`. THE SPAN IS THE
/// SCHEDULE: a state of span `j - i` reads only states of strictly shorter
/// span, so `bands` holds the states of each span — band position IS the
/// span — and a band's states are independent of one another.
///
/// @tparam Index The mesh's index type.
template <typename Index> struct hole_table {
  tf::buffer<Index> owned;
  tf::buffer<std::array<Index, 3>> facets;
  tf::buffer<tf::vector<double, 3>> normal;
  tf::buffer<double> area;
  tf::buffer<double> boundary;
  tf::buffer<char> retained;
  tf::offset_block_buffer<Index, std::array<Index, 4>> candidates;
  tf::buffer<std::array<Index, 3>> queries;
  tf::buffer<Index> block;
  tf::buffer<Index> cursor;
  tf::offset_block_buffer<Index, Index> bands;
  tf::buffer<double> bottleneck;
  tf::buffer<double> total;
  tf::buffer<Index> choice;
  tf::buffer<char> valid;
  tf::buffer<Index> stack;
};

/// The scratch one table-tier rim walks on: the tetrahedralization it is
/// protected against and solved over, and one provisional cycle expansion.
template <typename Index, typename Int, typename RealT>
struct hole_table_scratch {
  tf::delaunay_tetrahedralizer<Index, tf::fill::hole_site_coord<Int>, Int> dt;
  tf::buffer<Index> position_of_site;
  tf::topology::cdt::dt3::tet_edge_query_workspace<Index> edge_query;
  tf::buffer<Index> site_of_position, next_site_of_position;
  tf::buffer<tf::topology::cdt::dt3::tet_site_candidate<
      tf::fill::hole_site_coord<Int>>>
      site_candidates;
  tf::buffer<tf::exact::vertex<Index, tf::fill::hole_site_coord<Int>>>
      appended_sites;
  tf::buffer<Index> split_positions;
  tf::buffer<tf::exact::vertex<Index, tf::fill::hole_site_coord<Int>>> sites;
  tf::buffer<tf::point<RealT, 3>> positions;
  tf::buffer<tf::fill::hole_rim_edge<Index>> rim_edges;
  tf::buffer<Index> corners;
};

/// The band length below which a stage's states are solved serially, one
/// barrier being cheaper than a task for the many short bands a rim states.
inline constexpr unsigned long hole_table_band_grain = 4096;

} // namespace tf::fill

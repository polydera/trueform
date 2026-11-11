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
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../exact/pt_converter.hpp"
#include "../../topology/manifold_edge_link.hpp"
#include "./count_tet_open_edges.hpp"
#include <cstddef>

namespace tf::tetmesh {

/// The surface the entrance hands the mesher, which the drains append to
/// (@ref cut_tet_pairs): `points`, its lattice points on the `Int` lattice
/// `input_bits` wide; `triangles` over them, ordered facet by facet; `facets`,
/// the offsets of each facet's run of triangles; `link`, the entrance's
/// triangles' manifold edge link in that order; `parents`, the input face each
/// triangle cuts; the converter that maps the lattice back to the input's
/// coordinates; `band`, the door's band in lattice units, zero when no door ran
/// (@ref make_tet_entrance_surface); the triangles its statements threw out as
/// degenerate and added by subdividing their neighbours; and the edges the
/// entrance's triangles leave open (@ref count_tet_merged_open_edges). That
/// count is the entrance's, its verdict, open or closed, the surface's: a drain
/// re-tiles inside the boundary of the faces it retires, so it may split an
/// open edge but never opens or closes one. The build's weld of coincident
/// sites and the splits at sites standing on segments may close them, so an
/// open verdict is read again on the recovered surface.
template <typename Index, typename RealT, typename Int>
/// A facet's id names its tables and its block of `facets`.
struct tet_entrance_surface {
  using int_type = Int;
  tf::points_buffer<Int, 3> points;
  tf::blocked_buffer<Index, 3> triangles;
  tf::buffer<Index> facets;
  tf::manifold_edge_link<Index, 3> link;
  tf::buffer<Index> parents;
  tf::exact::pt_converter<Int, RealT, 3> converter;
  int input_bits = 0;
  Int band = Int(0);
  std::size_t dropped_degenerate = 0;
  std::size_t stitched_triangles = 0;
  tet_open_edges open_edges;
};

} // namespace tf::tetmesh

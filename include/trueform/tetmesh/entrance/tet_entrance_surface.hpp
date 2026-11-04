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
#include "./count_tet_open_edges.hpp"
#include "../../core/buffer.hpp"
#include "../../core/points_buffer.hpp"
#include "../../exact/pt_converter.hpp"
#include "../../topology/manifold_edge_link.hpp"
#include <cstddef>

namespace tf::tetmesh {

/// The surface the entrance hands the mesher: `points`, every lattice point
/// of the input's self-arrangement on the `Int` lattice `input_bits` wide;
/// `triangles`, the arrangement's faces over them, compatible by
/// construction, ordered facet by facet; `facets`, the offsets of each
/// facet's run of triangles; `link`, the triangles' manifold edge link in
/// that order; `parents`, the input face each triangle cuts; the converter
/// that maps the lattice back to the input's coordinates; the arrangement
/// `passes` it took; the triangles it threw out as degenerate and added by
/// subdividing their neighbours; the carriers and faces a pass left without
/// triangles, a surface with a hole stating nothing past them; and the edges
/// `triangles` leave open (@ref count_tet_open_edges).
template <typename Index, typename RealT, typename Int>
struct tet_entrance_surface {
  using int_type = Int;
  tf::points_buffer<Int, 3> points;
  tf::blocked_buffer<Index, 3> triangles;
  tf::buffer<Index> facets;
  tf::manifold_edge_link<Index, 3> link;
  tf::buffer<Index> parents;
  tf::exact::pt_converter<Int, RealT, 3> converter;
  int input_bits = 0;
  std::size_t passes = 0;
  std::size_t dropped_degenerate = 0;
  std::size_t stitched_triangles = 0;
  std::size_t failed_planes = 0;
  std::size_t failed_faces = 0;
  tet_open_edges open_edges;
};

} // namespace tf::tetmesh

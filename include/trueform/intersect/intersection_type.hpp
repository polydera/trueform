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
#pragma once
#include "../topology/topo_type.hpp"

namespace tf {

/// @ingroup intersect_types
/// @brief Enumeration of intersection topological types.
///
/// Describes how two geometric primitives intersect (vertex-vertex,
/// vertex-edge, edge-edge, etc.). See @ref tf::topo_type for the
/// primitive types used.
enum class intersection_type : char {
  vertex_vertex = 0,
  vertex_edge = 1,
  edge_vertex = 2,
  edge_edge = 3,
  vertex_face = 4,
  face_vertex = 5,
  edge_face = 6,
  face_edge = 7,
  none = 8
};

/// @ingroup intersect_types
/// @brief Create an intersection type from two topological types.
/// @param t0 The first @ref tf::topo_type.
/// @param t1 The second @ref tf::topo_type.
/// @return The corresponding @ref tf::intersection_type.
constexpr auto make_intersection_type(tf::topo_type t0, tf::topo_type t1)
    -> intersection_type {
  switch (static_cast<char>(t0) | (static_cast<char>(t1) << 3)) {
  case static_cast<char>(tf::topo_type::vertex) |
      (static_cast<char>(tf::topo_type::vertex) << 3):
    return intersection_type::vertex_vertex;
  case static_cast<char>(tf::topo_type::vertex) |
      (static_cast<char>(tf::topo_type::edge) << 3):
    return intersection_type::vertex_edge;
  case static_cast<char>(tf::topo_type::edge) |
      (static_cast<char>(tf::topo_type::vertex) << 3):
    return intersection_type::edge_vertex;
  case static_cast<char>(tf::topo_type::edge) |
      (static_cast<char>(tf::topo_type::edge) << 3):
    return intersection_type::edge_edge;
  case static_cast<char>(tf::topo_type::vertex) |
      (static_cast<char>(tf::topo_type::face) << 3):
    return intersection_type::vertex_face;
  case static_cast<char>(tf::topo_type::face) |
      (static_cast<char>(tf::topo_type::vertex) << 3):
    return intersection_type::face_vertex;
  case static_cast<char>(tf::topo_type::edge) |
      (static_cast<char>(tf::topo_type::face) << 3):
    return intersection_type::edge_face;
  case static_cast<char>(tf::topo_type::face) |
      (static_cast<char>(tf::topo_type::edge) << 3):
    return intersection_type::face_edge;
  default:
    return intersection_type::none;
  }
}
} // namespace tf

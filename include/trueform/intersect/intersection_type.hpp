/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../topology/topo_type.hpp"

namespace tf {
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

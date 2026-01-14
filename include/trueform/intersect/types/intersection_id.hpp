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
#include "../intersection_type.hpp"
#include <utility>

namespace tf::intersect {
template <typename Index> struct intersection_id {
  static constexpr Index polygon_tag = -1;
  static constexpr Index vertex_tag = -3;
  Index self_id0;
  Index self_id1;
  Index other_id0;
  Index other_id1;
  Index id;
  tf::intersection_type type;

  auto make_vertex_vertex(Index v0, Index v1, Index id) -> intersection_id {
    return {vertex_tag, v0, vertex_tag,
            v1,         id, tf::intersection_type::vertex_vertex};
  }

  auto make_vertex_edge(Index v0, Index e0, Index e1, Index id)
      -> intersection_id {
    if (e0 > e1)
      std::swap(e0, e1);
    return {vertex_tag, v0, e0, e1, id, tf::intersection_type::vertex_edge};
  }

  auto make_edge_vertex(Index e0, Index e1, Index v0, Index id)
      -> intersection_id {
    if (e0 > e1)
      std::swap(e0, e1);
    return {e0, e1, vertex_tag, v0, id, tf::intersection_type::edge_vertex};
  }

  auto make_edge_edge(Index e0, Index e1, Index e2, Index e3, Index id)
      -> intersection_id {
    if (e0 > e1)
      std::swap(e0, e1);
    if (e2 > e3)
      std::swap(e2, e3);
    return {e0, e1, e2, e3, id, tf::intersection_type::edge_edge};
  }

  auto make_vertex_face(Index v0, Index p1, Index id) -> intersection_id {
    return {vertex_tag, v0, polygon_tag,
            p1,         id, tf::intersection_type::vertex_face};
  }

  auto make_face_vertex(Index p0, Index v1, Index id) -> intersection_id {
    return {polygon_tag, p0, vertex_tag,
            v1,          id, tf::intersection_type::face_vertex};
  }

  auto make_edge_face(Index e0, Index e1, Index p0, Index id)
      -> intersection_id {
    if (e0 > e1)
      std::swap(e0, e1);
    return {e0, e1, polygon_tag, p0, id, tf::intersection_type::edge_face};
  }

  auto make_face_edge(Index p0, Index e0, Index e1, Index id)
      -> intersection_id {
    if (e0 > e1)
      std::swap(e0, e1);
    return {polygon_tag, p0, e0, e1, id, tf::intersection_type::face_edge};
  }

  friend auto operator==(const intersection_id &i0, const intersection_id &i1)
      -> bool {
    return std::make_pair(i0.self_id0, i0.self_id1, i0.other_id0,
                          i0.other_id1) ==
           std::make_pair(i0.self_id0, i0.self_id1, i0.other_id0, i1.other_id1);
  }
};
} // namespace tf::intersect

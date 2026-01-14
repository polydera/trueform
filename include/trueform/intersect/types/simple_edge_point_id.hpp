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
#include <tuple>
#include <utility>

namespace tf::intersect {
template <typename Index> struct simple_edge_point_id {
  Index vertex_id0;
  Index vertex_id1;
  Index point_id;
  simple_edge_point_id() = default;
  simple_edge_point_id(Index pt0, Index pt1, Index point_id)
      : vertex_id0{pt0}, vertex_id1{pt1}, point_id{point_id} {
    if (pt1 < pt0)
      std::swap(pt0, pt1);
  }

  friend auto operator<(const simple_edge_point_id &e0,
                        const simple_edge_point_id &e1) -> bool {
    return std::make_tuple(e0.vertex_id0 == e0.vertex_id1, e0.vertex_id0,
                           e0.vertex_id1) <
           std::make_tuple(e1.vertex_id0 == e1.vertex_id1, e1.vertex_id0,
                           e1.vertex_id1);
  }

  friend auto operator==(const simple_edge_point_id &e0,
                         const simple_edge_point_id &e1) -> bool {
    return std::make_pair(e0.vertex_id0, e0.vertex_id1) ==
           std::make_pair(e1.vertex_id0, e1.vertex_id1);
  }
};
} // namespace tf::intersect

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
#include <algorithm>
#include <utility>

namespace tf::intersect {
template <typename Index> struct simple_edge_point_id {
  Index vertex_id0;
  Index vertex_id1;
  Index cut;
  Index point_id;
  simple_edge_point_id() = default;
  simple_edge_point_id(Index pt0, Index pt1, Index cut, Index point_id)
      : vertex_id0{std::min(pt0, pt1)}, vertex_id1{std::max(pt0, pt1)},
        cut{cut}, point_id{point_id} {}

  friend auto operator<(const simple_edge_point_id &e0,
                        const simple_edge_point_id &e1) -> bool {
    return std::make_tuple(e0.vertex_id0 == e0.vertex_id1, e0.vertex_id0,
                           e0.vertex_id1, e0.cut) <
           std::make_tuple(e1.vertex_id0 == e1.vertex_id1, e1.vertex_id0,
                           e1.vertex_id1, e1.cut);
  }

  friend auto operator==(const simple_edge_point_id &e0,
                         const simple_edge_point_id &e1) -> bool {
    return std::make_tuple(e0.vertex_id0, e0.vertex_id1, e0.cut) ==
           std::make_tuple(e1.vertex_id0, e1.vertex_id1, e1.cut);
  }
};
} // namespace tf::intersect

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
#include "../../core/algorithm/circular_increment.hpp"
#include "../../core/intersects.hpp"

namespace tf::intersect::polygon {
template <typename Record, typename Handle0, typename Handle1>
auto vertex_edge(Record &&record, const Handle0 &handle0,
                 const Handle1 &handle1) {
  auto poly0_size = handle0.polygon.size();
  auto poly1_size = handle1.polygon.size();
  for (decltype(poly0_size) i = 0; i < poly0_size; i++) {
    auto next_i = tf::circular_increment(i, poly0_size);
    for (decltype(poly1_size) j = 0; j < poly1_size; j++) {
      if (handle0.representation.vertex[i] && handle1.representation.edge[j]) {
        auto next_j = tf::circular_increment(j, poly1_size);
        if (tf::intersects(handle0.polygon[i],
                           tf::make_segment_between_points(
                               handle1.polygon[j], handle1.polygon[next_j])))
          record(i, j, next_j, 0);
      }
      if (handle1.representation.vertex[j] && handle0.representation.edge[i]) {
        if (tf::intersects(handle1.polygon[j],
                           tf::make_segment_between_points(
                               handle0.polygon[i], handle0.polygon[next_i])))
          record(j, i, next_i, 1);
      }
    }
  }
}
} // namespace tf::intersect::polygon

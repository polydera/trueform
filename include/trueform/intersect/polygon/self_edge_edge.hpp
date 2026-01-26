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
#include "../../core/ray_hit.hpp"

namespace tf::intersect::polygon {
template <typename Record, typename Handle0, typename Handle1>
auto self_edge_edge(Record &&record, const Handle0 &handle0,
                    const Handle1 &handle1) {
  auto &&indices0 = handle0.polygon.indices();
  auto &&indices1 = handle1.polygon.indices();
  auto poly0_size = handle0.polygon.size();
  auto poly1_size = handle1.polygon.size();
  using RealT =
      tf::coordinate_type<decltype(handle0.polygon), decltype(handle1.polygon)>;
  for (decltype(poly0_size) i = 0; i < poly0_size; i++) {
    if (!handle0.representation.edge[i])
      continue;
    auto next_i = tf::circular_increment(i, poly0_size);
    auto ray0 = tf::make_ray_between_points(handle0.polygon[i],
                                            handle0.polygon[next_i]);
    for (decltype(poly0_size) j = 0; j < poly1_size; j++) {
      auto next_j = tf::circular_increment(j, poly1_size);
      if (!handle1.representation.edge[j])
        continue;
      if (indices0[i] == indices1[j] || indices0[i] == indices1[next_j] ||
          indices0[next_i] == indices1[j] ||
          indices0[next_i] == indices1[next_j])
        continue;
      auto edge1 = tf::make_segment_between_points(handle1.polygon[j],
                                                   handle1.polygon[next_j]);
      auto result =
          tf::ray_hit(ray0, edge1, tf::make_ray_config(RealT(0), RealT(1)));
      if (result) {
        record(i, next_i, j, next_j, result.point);
      }
    }
  }
}
} // namespace tf::intersect::polygon


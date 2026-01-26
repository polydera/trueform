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
auto edge_face(Record &&record, const Handle0 &handle0,
               const Handle1 &handle1) {
  using RealT =
      tf::coordinate_type<decltype(handle0.polygon), decltype(handle1.polygon)>;
  auto poly0_size = handle0.polygon.size();
  for (decltype(poly0_size) j = 0; j < poly0_size; j++) {
    if (!handle0.representation.edge[j])
      continue;
    auto next_j = tf::circular_increment(j, poly0_size);
    auto ray0 = tf::make_ray_between_points(handle0.polygon[j],
                                            handle0.polygon[next_j]);
    auto result = tf::ray_hit(ray0, handle1.polygon,
                              tf::make_ray_config(RealT(0), RealT(1)));
    if (result) {
      record(j, next_j, result.point, 0);
    }
  }

  auto poly1_size = handle1.polygon.size();
  for (decltype(poly1_size) j = 0; j < poly1_size; j++) {
    if (!handle1.representation.edge[j])
      continue;
    auto next_j = tf::circular_increment(j, poly1_size);
    auto ray1 = tf::make_ray_between_points(handle1.polygon[j],
                                            handle1.polygon[next_j]);
    auto result = tf::ray_hit(ray1, handle0.polygon,
                              tf::make_ray_config(RealT(0), RealT(1)));

    if (result) {
      record(j, next_j, result.point, 1);
    }
  }
}
} // namespace tf::intersect::polygon

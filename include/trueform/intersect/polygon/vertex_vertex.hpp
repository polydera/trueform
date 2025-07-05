/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/intersects.hpp"

namespace tf::intersect::polygon {

template <typename Record, typename Handle0, typename Handle1>
auto vertex_vertex(Record &&record, const Handle0 &handle0,
                   const Handle1 &handle1) {
  auto poly0_size = handle0.polygon.size();
  auto poly1_size = handle1.polygon.size();
  for (decltype(poly0_size) i = 0; i < poly0_size; i++) {
    if (!handle0.representation.vertex[i])
      continue;
    for (decltype(poly1_size) j = 0; j < poly1_size; j++) {
      if (!handle1.representation.vertex[j])
        continue;
      if (tf::intersects(handle0.polygon[i], handle1.polygon[j])) {
        record(i, j);
      }
    }
  }
}
} // namespace tf::intersect::polygon

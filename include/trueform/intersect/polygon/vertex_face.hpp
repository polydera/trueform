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
#include "../../core/intersects.hpp"

namespace tf::intersect::polygon {
template <typename Record, typename Handle0, typename Handle1>
auto vertex_face(Record &&record, const Handle0 &handle0,
                 const Handle1 &handle1) {
  auto poly1_size = handle1.polygon.size();
  for (decltype(poly1_size) j = 0; j < poly1_size; j++) {
    if (!handle1.representation.vertex[j])
      continue;
    if (tf::intersects(handle0.polygon, handle1.polygon[j])) {
      record(j, 1);
    }
  }
  auto poly0_size = handle0.polygon.size();
  for (decltype(poly0_size) j = 0; j < poly0_size; j++) {
    if (!handle0.representation.vertex[j])
      continue;
    if (tf::intersects(handle1.polygon, handle0.polygon[j])) {
      record(j, 0);
    }
  }
}
} // namespace tf::intersect::polygon

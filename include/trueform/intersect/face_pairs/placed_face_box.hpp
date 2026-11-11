/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/aabb.hpp"
#include "../../exact/vertex.hpp"

#include <cstddef>

namespace tf::intersect {

/// The exact box of a face's placed corners. The corners are the geometry
/// the kernels read, so this box holds the face with nothing to spare —
/// the descent pays for the door's motion, the block filter does not.
template <typename Index, typename Int>
auto placed_face_box(tf::exact::vertex_range<Index, Int> face)
    -> tf::aabb<Int, 3> {
  auto low = face[0].pt;
  auto high = low;
  for (std::size_t at = 1; at < face.size(); ++at)
    for (std::size_t k = 0; k < 3; ++k) {
      if (face[at].pt[k] < low[k])
        low[k] = face[at].pt[k];
      if (face[at].pt[k] > high[k])
        high[k] = face[at].pt[k];
    }
  return tf::make_aabb(low, high);
}

} // namespace tf::intersect

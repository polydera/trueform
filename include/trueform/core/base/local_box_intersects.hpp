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
#include "../epsilon.hpp"
#include <array>

namespace tf::core {

/// @brief Check if point in local coordinates is inside axis-aligned box
/// Box is [0, extent[0]] x [0, extent[1]] x ... x [0, extent[N-1]]
template <typename T, std::size_t N>
auto local_point_box_intersects(const std::array<T, N> &local_pt,
                                const std::array<T, N> &extent) -> bool {
  for (std::size_t i = 0; i < N; ++i) {
    if (local_pt[i] < -tf::epsilon<T> || local_pt[i] > extent[i] + tf::epsilon<T>)
      return false;
  }
  return true;
}

/// @brief Check if point in local coordinates is inside rectangle
/// Rectangle is [0, length[0]] x [0, length[1]] at z=0, with radius tolerance
template <typename T>
auto local_point_rectangle_intersects(const std::array<T, 3> &local_pt,
                                      const std::array<T, 2> &length,
                                      T radius) -> bool {
  // Check if within the rectangle bounds (with radius tolerance)
  if (local_pt[0] < -radius - tf::epsilon<T> ||
      local_pt[0] > length[0] + radius + tf::epsilon<T>)
    return false;
  if (local_pt[1] < -radius - tf::epsilon<T> ||
      local_pt[1] > length[1] + radius + tf::epsilon<T>)
    return false;
  // Check z distance is within radius
  if (local_pt[2] < -radius - tf::epsilon<T> ||
      local_pt[2] > radius + tf::epsilon<T>)
    return false;
  return true;
}

} // namespace tf::core

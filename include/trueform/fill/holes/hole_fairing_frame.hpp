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
#include "../../core/buffer.hpp"
#include "../../core/dot.hpp"
#include "../../core/point.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tf::fill {

/// The frame a stencil is solved in: its own box's centre, and its diagonal
/// as the unit of length. A box with no diagonal states no frame.
inline auto
hole_fairing_frame(const tf::buffer<tf::point<double, 3>> &positions,
                   tf::point<double, 3> &centre, double &scale) -> bool {
  if (!positions.size())
    return false;
  auto low = positions[0], high = positions[0];
  for (const auto &position : positions)
    for (std::size_t axis = 0; axis < 3; ++axis) {
      low[axis] = std::min(low[axis], position[axis]);
      high[axis] = std::max(high[axis], position[axis]);
    }
  const auto span = high - low;
  scale = std::sqrt(tf::dot(span, span));
  for (std::size_t axis = 0; axis < 3; ++axis)
    centre[axis] = 0.5 * (low[axis] + high[axis]);
  return std::isfinite(scale) && scale > 0.0;
}

} // namespace tf::fill

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

#include "trueform/core/small_vector.hpp"

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace tf::cpp::detail {

/// @brief The element count a shape names, refused where it cannot be one.
///
/// Every array a factory mints and every array a structural operation writes
/// states its extent this way, so the shape is proved once and the allocation
/// that follows is a size.
inline auto shape_size(const tf::small_vector<int, 3> &shape) -> std::size_t {
  if (shape.empty())
    throw std::invalid_argument("nd_array factory: shape must not be empty");

  std::size_t total = 1;
  for (const auto dimension : shape) {
    if (dimension < 0)
      throw std::invalid_argument(
          "nd_array factory: shape dimensions must be nonnegative");
    const auto value = static_cast<std::size_t>(dimension);
    if (value != 0 && total > std::numeric_limits<std::size_t>::max() / value)
      throw std::overflow_error("nd_array factory: shape size overflows");
    total *= value;
  }
  if (total > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::length_error("nd_array factory: size exceeds int range");
  return total;
}

/// @brief The axis a caller names, read against the shape it names it of.
///
/// The one producer of that fact for every operation that takes an axis. A
/// negative axis counts from the last, and an array with no axis to name has
/// none to answer with, so it is refused before the counting.
inline auto normalized_axis(const tf::small_vector<int, 3> &shape, int axis,
                            const char *operation) -> int {
  const auto dimensions = static_cast<int>(shape.size());
  if (dimensions == 0)
    throw std::invalid_argument(std::string(operation) +
                                ": array must have at least one axis");
  if (axis < 0)
    axis += dimensions;
  if (axis < 0 || axis >= dimensions)
    throw std::out_of_range(std::string(operation) + ": axis out of range");
  return axis;
}

} // namespace tf::cpp::detail

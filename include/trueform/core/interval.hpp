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
#include "./dot.hpp"
#include "./line_like.hpp"
#include "./polygon.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief 1D interval with min/max bounds.
///
/// Represents a closed interval [min, max] on a line.
///
/// @tparam T The numeric type.
template <typename T> struct interval {
  T min;
  T max;
};

namespace core {
template <typename Range, std::size_t Dims, typename Policy>
auto make_interval(const Range &r, const tf::line_like<Dims, Policy> &line) {
  std::decay_t<decltype(r[0][0])> low, high;
  low = high = tf::dot(r[0] - line.origin, line.direction);
  auto size = r.size();
  for (decltype(size) i = 1; i < size; i++) {
    decltype(low) tmp = tf::dot(r[i] - line.origin, line.direction);
    low = std::min(low, tmp);
    high = std::max(high, tmp);
  }
  return interval<decltype(low)>{low, high};
}
} // namespace core

/// @ingroup core_primitives
/// @brief Create interval by projecting polygon onto line.
///
/// Projects all vertices of the polygon onto the line and returns
/// the interval containing all projections.
///
/// @tparam Dims The coordinate dimensions.
/// @tparam Policy0 The polygon's policy type.
/// @tparam Policy1 The line's policy type.
/// @param poly The polygon to project.
/// @param line The line to project onto.
/// @return An @ref tf::interval containing min/max projection values.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto make_interval(const polygon<Dims, Policy0> &poly,
                   const tf::line_like<Dims, Policy1> &line) {
  return core::make_interval(poly, line);
}
} // namespace tf

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./point_like.hpp"
#include "./vector_like.hpp"
#include <cmath>

namespace tf {

/// @brief Find the axis with the largest absolute value in a vector_like
/// @param v The vector to examine
/// @return Index of the axis with largest |value|
template <std::size_t N, typename Policy>
auto largest_axis(const tf::vector_like<N, Policy>& v) {
  std::size_t max_axis = 0;
  auto abs_max = std::abs(v[0]);
  for (std::size_t i = 1; i < N; ++i) {
    auto abs_val = std::abs(v[i]);
    if (abs_val > abs_max) {
      abs_max = abs_val;
      max_axis = i;
    }
  }
  return max_axis;
}

/// @brief Find the axis with the largest absolute value in a point_like
/// @param p The point to examine
/// @return Index of the axis with largest |value|
template <std::size_t N, typename Policy>
auto largest_axis(const tf::point_like<N, Policy>& p) {
  std::size_t max_axis = 0;
  auto abs_max = std::abs(p[0]);
  for (std::size_t i = 1; i < N; ++i) {
    auto abs_val = std::abs(p[i]);
    if (abs_val > abs_max) {
      abs_max = abs_val;
      max_axis = i;
    }
  }
  return max_axis;
}

} // namespace tf

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

#include "./point_like.hpp"
#include "./vector_like.hpp"
#include <cmath>

namespace tf {

/// @ingroup core_queries
/// @brief Find the axis with the largest absolute component.
///
/// @tparam N The coordinate dimensions.
/// @tparam Policy The vector's policy type.
/// @param v The vector to examine.
/// @return Index of the axis with largest absolute value.
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

/// @ingroup core_queries
/// @brief Find the axis with the largest absolute coordinate.
/// @overload
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

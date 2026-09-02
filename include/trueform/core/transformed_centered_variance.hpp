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

#include "./coordinate_type.hpp"
#include "./linalg/is_identity.hpp"
#include "./transformation_like.hpp"

#include <cstddef>

namespace tf::core {

/// @brief Mean squared distance from a centroid, measured after a transform.
///
/// Computes (1/n) * sum ||t * (p_i - centroid)||^2, the variance in the space
/// @ref tf::core::transformed_cross_covariance puts its matrix in. Only the
/// linear part of `t` acts, a centered difference carrying no translation.
///
/// @param points The point set.
/// @param centroid The centroid of that point set, in its own space.
/// @param t Transformation applied to the point set.
/// @return The mean squared distance from the centroid, after `t`.
template <std::size_t Dims, typename U, typename Range, typename Point>
auto transformed_centered_variance(const Range &points, const Point &centroid,
                                   const transformation_like<Dims, U> &t) {
  using T = tf::coordinate_type<Point>;

  T sum = T(0);
  for (const auto &p : points) {
    if constexpr (tf::linalg::is_identity<U>) {
      for (std::size_t d = 0; d < Dims; ++d) {
        T diff = p[d] - centroid[d];
        sum += diff * diff;
      }
    } else {
      for (std::size_t i = 0; i < Dims; ++i) {
        T diff = T(0);
        for (std::size_t k = 0; k < Dims; ++k)
          diff += t(i, k) * (p[k] - centroid[k]);
        sum += diff * diff;
      }
    }
  }

  const auto n = points.size();
  return sum / T(n + (n == 0));
}

} // namespace tf::core

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

#include "./linalg/is_identity.hpp"
#include "./transformation_like.hpp"

#include <array>
#include <cstddef>

namespace tf::core {

/// @brief Transform a covariance matrix by a transformation.
///
/// Computes Cov' = M * Cov * M^T where only the linear part of M
/// is used (translations don't affect centered covariance).
///
/// Uses compile-time check to avoid unnecessary computation when
/// transformation is identity.
///
/// @param Cov The covariance matrix to transform.
/// @param t Transformation applied to the point set.
/// @return Transformed covariance matrix Cov' = M * Cov * M^T.
template <typename T, std::size_t Dims, typename U>
auto transformed_covariance(const std::array<std::array<T, Dims>, Dims> &Cov,
                            const transformation_like<Dims, U> &t) {
  constexpr bool is_id = tf::linalg::is_identity<U>;

  if constexpr (is_id) {
    return Cov;
  } else {
    // Cov' = M * Cov * M^T
    std::array<std::array<T, Dims>, Dims> temp{};
    for (std::size_t i = 0; i < Dims; ++i)
      for (std::size_t j = 0; j < Dims; ++j)
        for (std::size_t k = 0; k < Dims; ++k)
          temp[i][j] += t(i, k) * Cov[k][j];

    std::array<std::array<T, Dims>, Dims> out{};
    for (std::size_t i = 0; i < Dims; ++i)
      for (std::size_t j = 0; j < Dims; ++j)
        for (std::size_t k = 0; k < Dims; ++k)
          out[i][j] += temp[i][k] * t(j, k); // M^T[k][j] = M(j,k)
    return out;
  }
}

} // namespace tf::core

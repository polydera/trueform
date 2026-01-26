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

/// @brief Transform a cross-covariance matrix by two transformations.
///
/// Computes H' = tX * H * tY^T where only the rotation parts of tX and tY
/// are used (translations don't affect centered covariance).
///
/// Uses compile-time checks to avoid unnecessary computation when either
/// transformation is identity.
///
/// @param H The cross-covariance matrix to transform.
/// @param tX Transformation applied to X point set.
/// @param tY Transformation applied to Y point set.
/// @return Transformed cross-covariance matrix H' = tX * H * tY^T.
template <typename T, std::size_t Dims, typename UX, typename UY>
auto transformed_cross_covariance(const std::array<std::array<T, Dims>, Dims> &H,
                                  const transformation_like<Dims, UX> &tX,
                                  const transformation_like<Dims, UY> &tY) {
  constexpr bool is_idX = tf::linalg::is_identity<UX>;
  constexpr bool is_idY = tf::linalg::is_identity<UY>;

  if constexpr (is_idX && is_idY) {
    // Both identity: no transformation needed
    return H;
  } else if constexpr (!is_idX && !is_idY) {
    // Both transforms: H' = tX * H * tY^T
    std::array<std::array<T, Dims>, Dims> temp{};
    for (std::size_t i = 0; i < Dims; ++i)
      for (std::size_t j = 0; j < Dims; ++j)
        for (std::size_t k = 0; k < Dims; ++k)
          temp[i][j] += tX(i, k) * H[k][j];

    std::array<std::array<T, Dims>, Dims> out{};
    for (std::size_t i = 0; i < Dims; ++i)
      for (std::size_t j = 0; j < Dims; ++j)
        for (std::size_t k = 0; k < Dims; ++k)
          out[i][j] += temp[i][k] * tY(j, k); // tY^T[k][j] = tY(j,k)
    return out;
  } else if constexpr (!is_idX) {
    // Only X transform: H' = tX * H
    std::array<std::array<T, Dims>, Dims> out{};
    for (std::size_t i = 0; i < Dims; ++i)
      for (std::size_t j = 0; j < Dims; ++j)
        for (std::size_t k = 0; k < Dims; ++k)
          out[i][j] += tX(i, k) * H[k][j];
    return out;
  } else {
    // Only Y transform: H' = H * tY^T
    std::array<std::array<T, Dims>, Dims> out{};
    for (std::size_t i = 0; i < Dims; ++i)
      for (std::size_t j = 0; j < Dims; ++j)
        for (std::size_t k = 0; k < Dims; ++k)
          out[i][j] += H[i][k] * tY(j, k);
    return out;
  }
}

} // namespace tf::core

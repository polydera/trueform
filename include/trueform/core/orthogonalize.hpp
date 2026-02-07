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

#include "./transformation_like.hpp"
#include "./transformation.hpp"
#include "./sqrt.hpp"

namespace tf {

/// @ingroup core_linalg
/// @brief Orthogonalize the rotation part of a transformation in-place.
///
/// Uses Gram-Schmidt orthonormalization to ensure the rotation part
/// forms a proper orthonormal basis. This corrects numerical drift
/// that can accumulate when composing transformations.
///
/// @param T The transformation to orthogonalize (modified in-place).
template <std::size_t Dims, typename Policy>
auto orthogonalize(transformation_like<Dims, Policy> &T) -> void {
  static_assert(Dims == 2 || Dims == 3,
                "orthogonalize only supports 2D and 3D transformations");
  using Scalar = typename Policy::coordinate_type;

  if constexpr (Dims == 3) {
    // Column 0: normalize
    Scalar len0 = tf::sqrt(T(0, 0) * T(0, 0) + T(1, 0) * T(1, 0) +
                           T(2, 0) * T(2, 0));
    T(0, 0) /= len0;
    T(1, 0) /= len0;
    T(2, 0) /= len0;

    // Column 1: orthogonalize against column 0, then normalize
    Scalar dot01 = T(0, 0) * T(0, 1) + T(1, 0) * T(1, 1) + T(2, 0) * T(2, 1);
    T(0, 1) -= dot01 * T(0, 0);
    T(1, 1) -= dot01 * T(1, 0);
    T(2, 1) -= dot01 * T(2, 0);
    Scalar len1 = tf::sqrt(T(0, 1) * T(0, 1) + T(1, 1) * T(1, 1) +
                           T(2, 1) * T(2, 1));
    T(0, 1) /= len1;
    T(1, 1) /= len1;
    T(2, 1) /= len1;

    // Column 2: cross product of columns 0 and 1
    T(0, 2) = T(1, 0) * T(2, 1) - T(2, 0) * T(1, 1);
    T(1, 2) = T(2, 0) * T(0, 1) - T(0, 0) * T(2, 1);
    T(2, 2) = T(0, 0) * T(1, 1) - T(1, 0) * T(0, 1);
  } else if constexpr (Dims == 2) {
    // Column 0: normalize
    Scalar len0 = tf::sqrt(T(0, 0) * T(0, 0) + T(1, 0) * T(1, 0));
    T(0, 0) /= len0;
    T(1, 0) /= len0;

    // Column 1: perpendicular to column 0 (90° rotation)
    T(0, 1) = -T(1, 0);
    T(1, 1) = T(0, 0);
  }
}

/// @ingroup core_linalg
/// @brief Return an orthogonalized copy of a transformation.
///
/// Uses Gram-Schmidt orthonormalization to ensure the rotation part
/// forms a proper orthonormal basis. This corrects numerical drift
/// that can accumulate when composing transformations.
///
/// @param T The transformation to orthogonalize.
/// @return A new transformation with orthonormalized rotation.
template <std::size_t Dims, typename Policy>
auto orthogonalized(const transformation_like<Dims, Policy> &T)
    -> tf::transformation<typename Policy::coordinate_type, Dims> {
  static_assert(Dims == 2 || Dims == 3,
                "orthogonalized only supports 2D and 3D transformations");

  tf::transformation<typename Policy::coordinate_type, Dims> result = T;
  orthogonalize(result);
  return result;
}

} // namespace tf

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
#include "./linalg/trans.hpp"
#include "./vector_like.hpp"
#include "./coordinate_type.hpp"
#include "./transformation_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief An affine transformation matrix in N-dimensional space.
///
/// Represents an N x (N+1) affine transformation matrix that can encode
/// rotation, scaling, and translation. The last column stores the translation.
///
/// Factory functions include:
/// - @ref tf::make_identity_transformation() - identity matrix
/// - @ref tf::make_transformation_from_translation() - pure translation
/// - @ref tf::make_rotation() - rotation around axis or pivot
/// - @ref tf::make_rotation_aligning() - rotation aligning one direction to another
///
/// @tparam T The scalar type (e.g., float, double).
/// @tparam Dims The dimensionality (e.g., 2, 3).
template <typename T, std::size_t Dims>
using transformation =
    tf::transformation_like<Dims, tf::linalg::trans<T, Dims>>;

/// @ingroup core_primitives
/// @brief Create an identity transformation.
///
/// Returns a transformation that leaves all points unchanged.
///
/// @tparam T The scalar type.
/// @tparam Dims The dimensionality.
/// @return An identity @ref tf::transformation.
template <typename T, std::size_t Dims> auto make_identity_transformation() {
  tf::transformation<T, Dims> out;
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims + 1; ++j)
      out(i, j) = i == j;
  return out;
}

/// @ingroup core_primitives
/// @brief Create a translation transformation.
///
/// Returns a transformation that translates points by the given vector.
///
/// @tparam Dims The dimensionality.
/// @tparam T The vector policy type.
/// @param translation The translation vector.
/// @return A @ref tf::transformation encoding the translation.
template <std::size_t Dims, typename T>
auto make_transformation_from_translation(
    const tf::vector_like<Dims, T> &translation) {
  auto out = make_identity_transformation<tf::coordinate_type<T>, Dims>();
  for (std::size_t i = 0; i < Dims; ++i)
    out(i, Dims) = translation[i];
  return out;
}
} // namespace tf

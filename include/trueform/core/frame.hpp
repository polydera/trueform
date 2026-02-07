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

#include "./frame_like.hpp"
#include "./linalg/safe_frame.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief A coordinate frame with forward and inverse transformations.
///
/// Stores both a transformation and its inverse for efficient bidirectional
/// coordinate conversions. Used for attaching local coordinate systems to
/// geometric primitives.
///
/// Use @ref tf::make_frame() for construction.
///
/// @tparam T The scalar type (e.g., float, double).
/// @tparam Dims The dimensionality (e.g., 2, 3).
template <typename T, std::size_t Dims>
using frame = tf::frame_like<Dims, tf::linalg::safe_frame<T, Dims>>;

/// @ingroup core_primitives
/// @brief Create a frame from forward and inverse transformations.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy0 Policy type for the forward transformation.
/// @tparam Policy1 Policy type for the inverse transformation.
/// @param _transformation The forward transformation.
/// @param _inv_transformation The inverse transformation.
/// @return A @ref tf::frame instance.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto make_frame(
    const tf::transformation_like<Dims, Policy0> &_transformation,
    const tf::transformation_like<Dims, Policy1> &_inv_transformation) {
  return tf::frame<tf::coordinate_type<Policy0, Policy1>, Dims>{
      _transformation, _inv_transformation};
}

/// @ingroup core_primitives
/// @brief Create a frame from a transformation (inverse computed automatically).
///
/// @tparam Dims The dimensionality.
/// @tparam Policy Policy type for the transformation.
/// @param _transformation The forward transformation.
/// @return A @ref tf::frame instance with computed inverse.
template <std::size_t Dims, typename Policy>
auto make_frame(const tf::transformation_like<Dims, Policy> &_transformation) {
  return tf::frame<tf::coordinate_type<Policy>, Dims>{_transformation};
}

/// @ingroup core_primitives
/// @brief Create an identity frame.
///
/// @tparam T The scalar type (e.g., float, double).
/// @tparam Dims The dimensionality (e.g., 2, 3).
/// @return A @ref tf::frame instance representing identity.
template <typename T, std::size_t Dims>
auto make_identity_frame() {
  return tf::frame<T, Dims>{tf::make_identity_transformation<T, Dims>(),
                            tf::make_identity_transformation<T, Dims>()};
}
} // namespace tf

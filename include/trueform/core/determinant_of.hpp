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
#include "./frame_like.hpp"
#include "./transformation_like.hpp"
#include <cstddef>
#include <type_traits>

namespace tf {

/// @ingroup core_properties
/// @brief The determinant of a transformation's linear part.
template <std::size_t Dims, typename Policy>
auto determinant_of(const tf::transformation_like<Dims, Policy> &t)
    -> tf::coordinate_type<Policy> {
  static_assert(Dims == 2 || Dims == 3,
                "determinant_of states 2D and 3D transformations");
  if constexpr (Dims == 2)
    return t(0, 0) * t(1, 1) - t(0, 1) * t(1, 0);
  else
    return t(0, 0) * (t(1, 1) * t(2, 2) - t(1, 2) * t(2, 1)) -
           t(0, 1) * (t(1, 0) * t(2, 2) - t(1, 2) * t(2, 0)) +
           t(0, 2) * (t(1, 0) * t(2, 1) - t(1, 1) * t(2, 0));
}

/// @ingroup core_properties
/// @brief The identity's determinant: one.
/// @overload
template <typename T, std::size_t Dims>
constexpr auto determinant_of(const tf::identity_transformation<T, Dims> &)
    -> std::decay_t<T> {
  return std::decay_t<T>(1);
}

/// @ingroup core_properties
/// @brief The determinant of a frame's forward transformation.
/// @overload
template <std::size_t Dims, typename Policy>
auto determinant_of(const tf::frame_like<Dims, Policy> &f) {
  return tf::determinant_of(f.transformation());
}

/// @ingroup core_properties
/// @brief The identity frame's determinant: one.
/// @overload
template <typename T, std::size_t Dims>
constexpr auto determinant_of(const tf::identity_frame<T, Dims> &)
    -> std::decay_t<T> {
  return std::decay_t<T>(1);
}

} // namespace tf

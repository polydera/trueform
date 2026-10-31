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
#include "./determinant_of.hpp"
#include "./frame_like.hpp"
#include "./transformation_like.hpp"
#include <cstddef>
#include <type_traits>

namespace tf {

/// @ingroup core_properties
/// @brief The determinant of a transformation's rotation factor: `-1` when
///        it reflects, `1` otherwise. A positive scale never changes it.
template <std::size_t Dims, typename Policy>
auto determinant_of_rotation(const tf::transformation_like<Dims, Policy> &t)
    -> tf::coordinate_type<Policy> {
  using real_t = tf::coordinate_type<Policy>;
  return tf::determinant_of(t) < real_t(0) ? real_t(-1) : real_t(1);
}

/// @ingroup core_properties
/// @brief The identity's rotation: one.
/// @overload
template <typename T, std::size_t Dims>
constexpr auto
determinant_of_rotation(const tf::identity_transformation<T, Dims> &)
    -> std::decay_t<T> {
  return std::decay_t<T>(1);
}

/// @ingroup core_properties
/// @brief The rotation of a frame's forward transformation.
/// @overload
template <std::size_t Dims, typename Policy>
auto determinant_of_rotation(const tf::frame_like<Dims, Policy> &f) {
  return tf::determinant_of_rotation(f.transformation());
}

/// @ingroup core_properties
/// @brief The identity frame's rotation: one.
/// @overload
template <typename T, std::size_t Dims>
constexpr auto determinant_of_rotation(const tf::identity_frame<T, Dims> &)
    -> std::decay_t<T> {
  return std::decay_t<T>(1);
}

} // namespace tf

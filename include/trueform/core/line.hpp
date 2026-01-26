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
#include "./base/line.hpp"
#include "./line_like.hpp"
#include "./coordinate_type.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief An infinite line in N-dimensional space.
///
/// A line is defined by an origin point and a direction vector. It extends
/// infinitely in both directions along the direction vector. Supports
/// parametric evaluation: `line(t)` returns `origin + t * direction`.
///
/// Factory functions include:
/// - @ref tf::make_line() - from origin and direction (always owning)
/// - @ref tf::make_line_between_points() - from two points
/// - @ref tf::make_line_like() - creates view when inputs are views
///
/// @tparam T The scalar type (e.g., float, double).
/// @tparam Dims The dimensionality (e.g., 2, 3).
template <typename T, std::size_t Dims>
using line = tf::line_like<
    Dims, tf::core::line<Dims, tf::core::pt<T, Dims>, tf::core::vec<T, Dims>>>;

/// @ingroup core_primitives
/// @brief Constructs a line from an origin and a direction vector.
///
/// The direction vector defines the orientation of the line, which extends
/// infinitely in both directions. The resulting line's value type is deduced
/// from the common type of the origin and direction components.
///
/// @tparam T0 The type of the origin vector's components.
/// @tparam T1 The type of the direction vector's components.
/// @tparam Dims The spatial dimension of the vectors.
/// @param origin A point on the line.
/// @param direction The direction of the line (not necessarily normalized).
/// @return A line object.
template <typename T0, std::size_t Dims, typename T1>
auto make_line(const tf::point_like<Dims, T0> &origin,
               const tf::vector_like<Dims, T1> &direction) {
  return line<tf::coordinate_type<T0, T1>, Dims>{origin, direction};
}

/// @ingroup core_primitives
/// @brief Constructs a line from two points.
///
/// The direction of the line is computed as `end - origin`. The line passes
/// through both `origin` and `end`, and extends infinitely in both directions.
///
/// @tparam T0 The type of the origin point.
/// @tparam T1 The type of the end point.
/// @tparam Dims The spatial dimension of the points.
/// @param origin The first point on the line.
/// @param end The second point on the line (used to compute the direction).
/// @return A line object.
template <typename T0, std::size_t Dims, typename T1>
auto make_line_between_points(const tf::point_like<Dims, T0> &origin,
                              const tf::point_like<Dims, T1> &end) {
  return line<tf::coordinate_type<T0, T1>, Dims>{origin, end - origin};
}

} // namespace tf

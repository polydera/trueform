/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./cross.hpp"
#include "./point_like.hpp"
#include "./unit_vector.hpp"

namespace tf {
/// @ingroup geometry
/// @brief Computes a unit normal vector from three points in 3D.
///
/// The resulting normal vector is computed using the cross product of the
/// vectors formed by the input points: `(pt2 - pt0) × (pt0 - pt1)`. The result
/// is then normalized using `tf::make_unit_vector`.
///
/// This is useful for computing face normals of triangles in mesh geometry,
/// where the winding order determines the direction of the normal (right-hand
/// rule).
///
/// @tparam T0 Type of the first point.
/// @tparam T1 Type of the second point.
/// @tparam T2 Type of the third point.
/// @param pt0 First point.
/// @param pt1 Second point.
/// @param pt2 Third point.
/// @return A `unit_vector<common_value<T0, T1, T2>, 3>` representing the unit
/// normal.
template <typename T0, typename T1, typename T2>
auto make_normal(const point_like<3, T0> &pt0, const point_like<3, T1> &pt1,
                 const point_like<3, T2> &pt2) {
  return tf::make_unit_vector(tf::cross(pt2 - pt0, pt0 - pt1));
}

} // namespace tf

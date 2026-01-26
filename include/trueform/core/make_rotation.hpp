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
#include "./angle.hpp"
#include "./axis.hpp"
#include "./cross.hpp"
#include "./dot.hpp"
#include "./epsilon.hpp"
#include "./normalized.hpp"
#include "./point_like.hpp"
#include "./transformation.hpp"
#include "./unit_vector_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Create 3D rotation around arbitrary axis.
///
/// @tparam T The scalar type.
/// @tparam Policy The axis unit vector's policy type.
/// @param angle The rotation angle.
/// @param axis The unit vector defining the rotation axis.
/// @return A 3D @ref tf::transformation.
template <typename T, typename Policy>
auto make_rotation(rad<T> angle, const unit_vector_like<3, Policy>& axis)
    -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  T t = T{1} - c;
  T x = axis[0], y = axis[1], z = axis[2];
  return transformation<T, 3>{
    t*x*x + c,     t*x*y - s*z,   t*x*z + s*y,   0,
    t*x*y + s*z,   t*y*y + c,     t*y*z - s*x,   0,
    t*x*z - s*y,   t*y*z + s*x,   t*z*z + c,     0
  };
}

/// @overload
template <typename T, typename Policy>
auto make_rotation(deg<T> angle, const unit_vector_like<3, Policy>& axis)
    -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis);
}

/// @ingroup core_primitives
/// @brief Create 3D rotation around X axis.
///
/// Optimized version for rotation around the X principal axis.
///
/// @tparam T The scalar type.
/// @param angle The rotation angle.
/// @return A 3D @ref tf::transformation.
template <typename T>
auto make_rotation(rad<T> angle, axis_t<0>) -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  return transformation<T, 3>{
    1, 0,  0, 0,
    0, c, -s, 0,
    0, s,  c, 0
  };
}

/// @overload
template <typename T>
auto make_rotation(deg<T> angle, axis_t<0>) -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis_t<0>{});
}

/// @ingroup core_primitives
/// @brief Create 3D rotation around Y axis.
/// @overload
template <typename T>
auto make_rotation(rad<T> angle, axis_t<1>) -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  return transformation<T, 3>{
     c, 0, s, 0,
     0, 1, 0, 0,
    -s, 0, c, 0
  };
}

/// @overload
template <typename T>
auto make_rotation(deg<T> angle, axis_t<1>) -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis_t<1>{});
}

/// @ingroup core_primitives
/// @brief Create 3D rotation around Z axis.
/// @overload
template <typename T>
auto make_rotation(rad<T> angle, axis_t<2>) -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  return transformation<T, 3>{
    c, -s, 0, 0,
    s,  c, 0, 0,
    0,  0, 1, 0
  };
}

/// @overload
template <typename T>
auto make_rotation(deg<T> angle, axis_t<2>) -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis_t<2>{});
}

/// @ingroup core_primitives
/// @brief Create 2D rotation.
///
/// @tparam T The scalar type.
/// @param angle The rotation angle.
/// @return A 2D @ref tf::transformation.
template <typename T>
auto make_rotation(rad<T> angle) -> transformation<T, 2> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  return transformation<T, 2>{
    c, -s, 0,
    s,  c, 0
  };
}

/// @overload
template <typename T>
auto make_rotation(deg<T> angle) -> transformation<T, 2> {
  return make_rotation(rad<T>{angle});
}

/// @ingroup core_primitives
/// @brief Create 3D rotation around axis through pivot point.
///
/// @tparam T The scalar type.
/// @tparam AxisPolicy The axis unit vector's policy type.
/// @tparam PointPolicy The pivot point's policy type.
/// @param angle The rotation angle.
/// @param axis The unit vector defining the rotation axis.
/// @param pivot The point the axis passes through.
/// @return A 3D @ref tf::transformation.
template <typename T, typename AxisPolicy, typename PointPolicy>
auto make_rotation(rad<T> angle, const unit_vector_like<3, AxisPolicy>& axis,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  T t = T{1} - c;
  T x = axis[0], y = axis[1], z = axis[2];
  T px = pivot[0], py = pivot[1], pz = pivot[2];
  T r00 = t*x*x + c,   r01 = t*x*y - s*z, r02 = t*x*z + s*y;
  T r10 = t*x*y + s*z, r11 = t*y*y + c,   r12 = t*y*z - s*x;
  T r20 = t*x*z - s*y, r21 = t*y*z + s*x, r22 = t*z*z + c;
  return transformation<T, 3>{
    r00, r01, r02, px - (r00*px + r01*py + r02*pz),
    r10, r11, r12, py - (r10*px + r11*py + r12*pz),
    r20, r21, r22, pz - (r20*px + r21*py + r22*pz)
  };
}

/// @overload
template <typename T, typename AxisPolicy, typename PointPolicy>
auto make_rotation(deg<T> angle, const unit_vector_like<3, AxisPolicy>& axis,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis, pivot);
}

/// @ingroup core_primitives
/// @brief Create 3D rotation around X axis through pivot point.
/// @overload
template <typename T, typename PointPolicy>
auto make_rotation(rad<T> angle, axis_t<0>,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  T t = T{1} - c;
  T py = pivot[1], pz = pivot[2];
  return transformation<T, 3>{
    1, 0,  0, 0,
    0, c, -s, py*t + s*pz,
    0, s,  c, pz*t - s*py
  };
}

/// @overload
template <typename T, typename PointPolicy>
auto make_rotation(deg<T> angle, axis_t<0>,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis_t<0>{}, pivot);
}

/// @ingroup core_primitives
/// @brief Create 3D rotation around Y axis through pivot point.
/// @overload
template <typename T, typename PointPolicy>
auto make_rotation(rad<T> angle, axis_t<1>,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  T t = T{1} - c;
  T px = pivot[0], pz = pivot[2];
  return transformation<T, 3>{
     c, 0, s, px*t - s*pz,
     0, 1, 0, 0,
    -s, 0, c, pz*t + s*px
  };
}

/// @overload
template <typename T, typename PointPolicy>
auto make_rotation(deg<T> angle, axis_t<1>,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis_t<1>{}, pivot);
}

/// @ingroup core_primitives
/// @brief Create 3D rotation around Z axis through pivot point.
/// @overload
template <typename T, typename PointPolicy>
auto make_rotation(rad<T> angle, axis_t<2>,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  T t = T{1} - c;
  T px = pivot[0], py = pivot[1];
  return transformation<T, 3>{
    c, -s, 0, px*t + s*py,
    s,  c, 0, py*t - s*px,
    0,  0, 1, 0
  };
}

/// @overload
template <typename T, typename PointPolicy>
auto make_rotation(deg<T> angle, axis_t<2>,
                   const point_like<3, PointPolicy>& pivot)
    -> transformation<T, 3> {
  return make_rotation(rad<T>{angle}, axis_t<2>{}, pivot);
}

/// @ingroup core_primitives
/// @brief Create 2D rotation around pivot point.
///
/// @tparam T The scalar type.
/// @tparam PointPolicy The pivot point's policy type.
/// @param angle The rotation angle.
/// @param pivot The center of rotation.
/// @return A 2D @ref tf::transformation.
template <typename T, typename PointPolicy>
auto make_rotation(rad<T> angle, const point_like<2, PointPolicy>& pivot)
    -> transformation<T, 2> {
  T c = tf::cos(angle);
  T s = tf::sin(angle);
  T t = T{1} - c;
  T px = pivot[0], py = pivot[1];
  return transformation<T, 2>{
    c, -s, px*t + s*py,
    s,  c, py*t - s*px
  };
}

/// @overload
template <typename T, typename PointPolicy>
auto make_rotation(deg<T> angle, const point_like<2, PointPolicy>& pivot)
    -> transformation<T, 2> {
  return make_rotation(rad<T>{angle}, pivot);
}

/// @ingroup core_primitives
/// @brief Create 2D rotation aligning one direction to another.
///
/// Returns a transformation that rotates the `from` direction to the `to`
/// direction.
///
/// @tparam Policy0 The policy type of the first unit vector.
/// @tparam Policy1 The policy type of the second unit vector.
/// @param from The source direction (unit vector).
/// @param to The target direction (unit vector).
/// @return A rotation transformation aligning `from` to `to`.
template <typename Policy0, typename Policy1>
auto make_rotation_aligning(const unit_vector_like<2, Policy0>& from,
                            const unit_vector_like<2, Policy1>& to)
    -> transformation<tf::coordinate_type<Policy0, Policy1>, 2> {
  using T = tf::coordinate_type<Policy0, Policy1>;

  // 2D cross product gives signed scalar
  T cross = from[0] * to[1] - from[1] * to[0];
  T dot = from[0] * to[0] + from[1] * to[1];
  auto angle = rad<T>{std::atan2(cross, dot)};
  return make_rotation(angle);
}

/// @ingroup core_primitives
/// @brief Create 3D rotation aligning one direction to another.
///
/// Returns a transformation that rotates the `from` direction to the `to`
/// direction. Handles the edge cases of parallel and anti-parallel vectors.
///
/// @tparam Policy0 The policy type of the first unit vector.
/// @tparam Policy1 The policy type of the second unit vector.
/// @param from The source direction (unit vector).
/// @param to The target direction (unit vector).
/// @return A rotation transformation aligning `from` to `to`.
template <typename Policy0, typename Policy1>
auto make_rotation_aligning(const unit_vector_like<3, Policy0>& from,
                            const unit_vector_like<3, Policy1>& to)
    -> transformation<tf::coordinate_type<Policy0, Policy1>, 3> {
  using T = tf::coordinate_type<Policy0, Policy1>;

  T d = tf::dot(from, to);

  // Parallel vectors (same direction) - return identity
  if (d > T{1} - tf::epsilon<T>) {
    return make_identity_transformation<T, 3>();
  }

  // Anti-parallel vectors - rotate 180° around any perpendicular axis
  if (d < T{-1} + tf::epsilon<T>) {
    // Find a perpendicular axis by crossing with a non-parallel basis vector
    auto axis_candidate = tf::cross(from, make_unit_vector<T, 3>(tf::axis<0>));
    if (axis_candidate.length() < tf::epsilon<T>) {
      axis_candidate = tf::cross(from, make_unit_vector<T, 3>(tf::axis<1>));
    }
    auto axis = tf::normalized(axis_candidate);
    return make_rotation(rad<T>{pi<T>}, axis);
  }

  // General case: rotate around cross product axis
  auto axis = tf::normalized(tf::cross(from, to));
  auto angle = rad<T>{std::acos(d)};
  return make_rotation(angle, axis);
}

} // namespace tf

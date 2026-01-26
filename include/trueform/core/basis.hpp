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
#include "./cross.hpp"
#include "./dot.hpp"
#include "./plane_like.hpp"
#include "./point_like.hpp"
#include "./unit_vector.hpp"
#include "./vector.hpp"

namespace tf {

/// @ingroup core_properties
/// @brief Construct orthonormal basis from a normal vector.
///
/// Returns two unit vectors orthogonal to the normal and to each other.
///
/// @tparam Policy The policy type of the normal.
/// @param normal The input unit normal vector.
/// @return Array of two @ref tf::unit_vector orthogonal to the normal.
template <typename Policy>
auto make_basis_from_normal(const unit_vector_like<3, Policy> &normal) {
  tf::vector<tf::coordinate_type<Policy>, 3> t0;
  if (std::abs(normal[0]) < std::abs(normal[1])) {
    t0[0] = 0;
    t0[1] = -normal[2];
    t0[2] = normal[1];
  } else {
    t0[0] = normal[2];
    t0[1] = 0;
    t0[2] = -normal[0];
  }
  auto t1 = tf::cross(normal, t0);
  return std::array<tf::unit_vector<tf::coordinate_type<Policy>, 3>, 2>{t0, t1};
}

/// @ingroup core_properties
/// @brief Construct orthonormal basis from a plane.
///
/// Returns two unit vectors tangent to the plane, orthogonal to
/// the normal and to each other.
///
/// @tparam Policy The policy type of the plane.
/// @param plane The input @ref tf::plane_like.
/// @return Array of two @ref tf::unit_vector tangent to the plane.
template <typename Policy> auto make_basis(const plane_like<3, Policy> &plane) {
  return make_basis_from_normal(plane.normal);
}

/// @ingroup core_properties
/// @brief Construct orthonormal basis from three points.
///
/// Creates basis vectors from p0→p1 (normalized) and the
/// component of p0→p2 orthogonal to the first.
///
/// @tparam Dims The coordinate dimensions.
/// @param p0 First point (origin).
/// @param p1 Second point (defines first basis vector direction).
/// @param p2 Third point (defines plane for second basis vector).
/// @return Array of two @ref tf::unit_vector.
template <std::size_t Dims, typename Policy0, typename Policy1,
          typename Policy2>
auto make_basis(const tf::point_like<Dims, Policy0> &p0,
                const tf::point_like<Dims, Policy1> &p1,
                const tf::point_like<Dims, Policy2> &p2) {
  using real_t = tf::coordinate_type<Policy0, Policy1, Policy2>;
  tf::unit_vector<real_t, Dims> e0 = p1 - p0;
  auto e1 = p2 - p0;
  auto cd0 = tf::dot(e0, e1) * e0;
  e1 -= cd0;
  return std::array<tf::unit_vector<real_t, Dims>, 2>{e0, e1};
}
} // namespace tf

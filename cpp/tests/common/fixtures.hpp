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

#include "carriers.hpp"
#include "nd_array.hpp"

#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/geometry/make_sphere_mesh.hpp"

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace tf::cpp::test {

template <typename Real>
auto triangle_mesh() -> owned_mesh<tf::cpp::default_index_t, Real> {
  static_assert(std::is_floating_point<Real>::value,
                "mesh fixtures require a floating-point coordinate type");
  return {polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2}, {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0},
                  Real{1}, Real{0}})};
}

template <typename Real>
auto two_triangle_mesh() -> owned_mesh<tf::cpp::default_index_t, Real> {
  return {polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 0, 2, 3},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{1}, Real{1},
       Real{0}, Real{0}, Real{1}, Real{0}})};
}

template <typename Real>
auto tetrahedron_mesh() -> owned_mesh<tf::cpp::default_index_t, Real> {
  return {polygons_of<tf::cpp::default_index_t, Real>(
      {0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{1},
       Real{0}, Real{0}, Real{0}, Real{1}})};
}

template <typename Real>
auto box_mesh() -> owned_mesh<tf::cpp::default_index_t, Real> {
  return {polygons_of<tf::cpp::default_index_t, Real>(
      {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
       3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5},
      {Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{1}, Real{1},
       Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{1}, Real{1},
       Real{0}, Real{1}, Real{1}, Real{1}, Real{1}, Real{0}, Real{1},
       Real{1}})};
}

/// A deterministic facade-produced sphere with 20 points and 36 faces.
template <typename Real>
auto sphere_mesh() -> owned_mesh<tf::cpp::default_index_t, Real> {
  static_assert(std::is_floating_point<Real>::value,
                "mesh fixtures require a floating-point coordinate type");
  return {tf::cpp::make_sphere_mesh(Real{1}, 4, 6)};
}

template <typename Real>
auto plane_mesh() -> owned_mesh<tf::cpp::default_index_t, Real> {
  return {polygons_of<tf::cpp::default_index_t, Real>(
      {0, 1, 2, 0, 2, 3},
      {Real{-1}, Real{-1}, Real{0}, Real{1}, Real{-1}, Real{0}, Real{1},
       Real{1}, Real{0}, Real{-1}, Real{1}, Real{0}})};
}

template <typename Real>
auto empty_mesh() -> owned_mesh<tf::cpp::default_index_t, Real> {
  return {};
}

template <typename Real> auto point_cloud_fixture() -> owned_point_cloud<Real> {
  return {points_of<Real>({Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0},
                           Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{1},
                           Real{1}, Real{1}, Real{1}})};
}

template <typename Real> auto identity_matrix() -> tf::cpp::nd_array<Real> {
  return make_nd_array<Real>(
      {Real{1}, Real{0}, Real{0}, Real{0}, Real{0}, Real{1}, Real{0}, Real{0},
       Real{0}, Real{0}, Real{1}, Real{0}, Real{0}, Real{0}, Real{0}, Real{1}},
      {4, 4});
}

template <typename Real>
auto translation_matrix(Real x, Real y, Real z) -> tf::cpp::nd_array<Real> {
  return make_nd_array<Real>({Real{1}, Real{0}, Real{0}, x, Real{0}, Real{1},
                              Real{0}, y, Real{0}, Real{0}, Real{1}, z, Real{0},
                              Real{0}, Real{0}, Real{1}},
                             {4, 4});
}

template <typename Real>
auto rotation_x_matrix(Real radians) -> tf::cpp::nd_array<Real> {
  const auto cosine = static_cast<Real>(std::cos(radians));
  const auto sine = static_cast<Real>(std::sin(radians));
  return make_nd_array<Real>({Real{1}, Real{0}, Real{0}, Real{0}, Real{0},
                              cosine, -sine, Real{0}, Real{0}, sine, cosine,
                              Real{0}, Real{0}, Real{0}, Real{0}, Real{1}},
                             {4, 4});
}

template <typename Real>
auto rotation_y_matrix(Real radians) -> tf::cpp::nd_array<Real> {
  const auto cosine = static_cast<Real>(std::cos(radians));
  const auto sine = static_cast<Real>(std::sin(radians));
  return make_nd_array<Real>({cosine, Real{0}, sine, Real{0}, Real{0}, Real{1},
                              Real{0}, Real{0}, -sine, Real{0}, cosine, Real{0},
                              Real{0}, Real{0}, Real{0}, Real{1}},
                             {4, 4});
}

template <typename Real>
auto rotation_z_matrix(Real radians) -> tf::cpp::nd_array<Real> {
  const auto cosine = static_cast<Real>(std::cos(radians));
  const auto sine = static_cast<Real>(std::sin(radians));
  return make_nd_array<Real>({cosine, -sine, Real{0}, Real{0}, sine, cosine,
                              Real{0}, Real{0}, Real{0}, Real{0}, Real{1},
                              Real{0}, Real{0}, Real{0}, Real{0}, Real{1}},
                             {4, 4});
}

template <typename Real>
auto scale_matrix(Real x, Real y, Real z) -> tf::cpp::nd_array<Real> {
  return make_nd_array<Real>({x, Real{0}, Real{0}, Real{0}, Real{0}, y, Real{0},
                              Real{0}, Real{0}, Real{0}, z, Real{0}, Real{0},
                              Real{0}, Real{0}, Real{1}},
                             {4, 4});
}

template <typename Real>
auto scale_matrix(Real scale) -> tf::cpp::nd_array<Real> {
  return scale_matrix(scale, scale, scale);
}

} // namespace tf::cpp::test

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
#include "./dot.hpp"
#include "./epsilon.hpp"
#include "./obb_like.hpp"

namespace tf::core {

namespace detail {

template <typename T, typename Policy0, typename Policy1>
auto obb_intersects_obb_2d(const tf::obb_like<2, Policy0> &obb0,
                           const tf::obb_like<2, Policy1> &obb1) -> bool {
  constexpr T reps = tf::epsilon<T>;

  // Compute centers from corner + half extents
  auto center0 = obb0.origin + obb0.axes[0] * (obb0.extent[0] * T(0.5)) +
                 obb0.axes[1] * (obb0.extent[1] * T(0.5));
  auto center1 = obb1.origin + obb1.axes[0] * (obb1.extent[0] * T(0.5)) +
                 obb1.axes[1] * (obb1.extent[1] * T(0.5));

  std::array<T, 2> a{obb0.extent[0] * T(0.5), obb0.extent[1] * T(0.5)};
  std::array<T, 2> b{obb1.extent[0] * T(0.5), obb1.extent[1] * T(0.5)};

  // Rotation matrix: rot[i][j] = dot(obb0.axes[i], obb1.axes[j])
  std::array<std::array<T, 2>, 2> rot;
  std::array<std::array<T, 2>, 2> rot_abs;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      rot[i][j] = tf::dot(obb0.axes[i], obb1.axes[j]);
      rot_abs[i][j] = std::abs(rot[i][j]) + reps;
    }
  }

  // Translation vector in obb0's frame
  auto diff = center1 - center0;
  std::array<T, 2> t_vec{tf::dot(diff, obb0.axes[0]), tf::dot(diff, obb0.axes[1])};

  T t, s;

  // Test axes A0, A1 (edge normals of obb0)
  // A0
  t = std::abs(t_vec[0]);
  if (t > a[0] + rot_abs[0][0] * b[0] + rot_abs[0][1] * b[1])
    return false;

  // A1
  t = std::abs(t_vec[1]);
  if (t > a[1] + rot_abs[1][0] * b[0] + rot_abs[1][1] * b[1])
    return false;

  // Test axes B0, B1 (edge normals of obb1)
  // B0
  s = rot[0][0] * t_vec[0] + rot[1][0] * t_vec[1];
  t = std::abs(s);
  if (t > b[0] + rot_abs[0][0] * a[0] + rot_abs[1][0] * a[1])
    return false;

  // B1
  s = rot[0][1] * t_vec[0] + rot[1][1] * t_vec[1];
  t = std::abs(s);
  if (t > b[1] + rot_abs[0][1] * a[0] + rot_abs[1][1] * a[1])
    return false;

  return true;
}

template <typename T, typename Policy0, typename Policy1>
auto obb_intersects_obb_3d(const tf::obb_like<3, Policy0> &obb0,
                           const tf::obb_like<3, Policy1> &obb1) -> bool {
  constexpr T reps = tf::epsilon<T>;

  auto center0 = obb0.origin + obb0.axes[0] * (obb0.extent[0] * T(0.5)) +
                 obb0.axes[1] * (obb0.extent[1] * T(0.5)) +
                 obb0.axes[2] * (obb0.extent[2] * T(0.5));
  auto center1 = obb1.origin + obb1.axes[0] * (obb1.extent[0] * T(0.5)) +
                 obb1.axes[1] * (obb1.extent[1] * T(0.5)) +
                 obb1.axes[2] * (obb1.extent[2] * T(0.5));

  std::array<T, 3> a{obb0.extent[0] * T(0.5), obb0.extent[1] * T(0.5),
                     obb0.extent[2] * T(0.5)};
  std::array<T, 3> b{obb1.extent[0] * T(0.5), obb1.extent[1] * T(0.5),
                     obb1.extent[2] * T(0.5)};

  std::array<std::array<T, 3>, 3> rot;
  std::array<std::array<T, 3>, 3> rot_abs;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      rot[i][j] = tf::dot(obb0.axes[i], obb1.axes[j]);
      rot_abs[i][j] = std::abs(rot[i][j]) + reps;
    }
  }

  // Translation vector in obb0's frame
  auto diff = center1 - center0;
  std::array<T, 3> t_vec;
  for (int i = 0; i < 3; ++i) {
    t_vec[i] = tf::dot(diff, obb0.axes[i]);
  }

  T t, s;

  // Test axes A0, A1, A2 (face normals of obb0)
  // A0
  t = std::abs(t_vec[0]);
  if (t >
      a[0] + rot_abs[0][0] * b[0] + rot_abs[0][1] * b[1] + rot_abs[0][2] * b[2])
    return false;

  // A1
  t = std::abs(t_vec[1]);
  if (t >
      a[1] + rot_abs[1][0] * b[0] + rot_abs[1][1] * b[1] + rot_abs[1][2] * b[2])
    return false;

  // A2
  t = std::abs(t_vec[2]);
  if (t >
      a[2] + rot_abs[2][0] * b[0] + rot_abs[2][1] * b[1] + rot_abs[2][2] * b[2])
    return false;

  // Test axes B0, B1, B2 (face normals of obb1)
  // B0
  s = rot[0][0] * t_vec[0] + rot[1][0] * t_vec[1] + rot[2][0] * t_vec[2];
  t = std::abs(s);
  if (t >
      b[0] + rot_abs[0][0] * a[0] + rot_abs[1][0] * a[1] + rot_abs[2][0] * a[2])
    return false;

  // B1
  s = rot[0][1] * t_vec[0] + rot[1][1] * t_vec[1] + rot[2][1] * t_vec[2];
  t = std::abs(s);
  if (t >
      b[1] + rot_abs[0][1] * a[0] + rot_abs[1][1] * a[1] + rot_abs[2][1] * a[2])
    return false;

  // B2
  s = rot[0][2] * t_vec[0] + rot[1][2] * t_vec[1] + rot[2][2] * t_vec[2];
  t = std::abs(s);
  if (t >
      b[2] + rot_abs[0][2] * a[0] + rot_abs[1][2] * a[1] + rot_abs[2][2] * a[2])
    return false;

  // Test 9 edge cross products (A_i x B_j)
  // A0 x B0
  s = t_vec[2] * rot[1][0] - t_vec[1] * rot[2][0];
  t = std::abs(s);
  if (t > a[1] * rot_abs[2][0] + a[2] * rot_abs[1][0] + b[1] * rot_abs[0][2] +
              b[2] * rot_abs[0][1])
    return false;

  // A0 x B1
  s = t_vec[2] * rot[1][1] - t_vec[1] * rot[2][1];
  t = std::abs(s);
  if (t > a[1] * rot_abs[2][1] + a[2] * rot_abs[1][1] + b[0] * rot_abs[0][2] +
              b[2] * rot_abs[0][0])
    return false;

  // A0 x B2
  s = t_vec[2] * rot[1][2] - t_vec[1] * rot[2][2];
  t = std::abs(s);
  if (t > a[1] * rot_abs[2][2] + a[2] * rot_abs[1][2] + b[0] * rot_abs[0][1] +
              b[1] * rot_abs[0][0])
    return false;

  // A1 x B0
  s = t_vec[0] * rot[2][0] - t_vec[2] * rot[0][0];
  t = std::abs(s);
  if (t > a[0] * rot_abs[2][0] + a[2] * rot_abs[0][0] + b[1] * rot_abs[1][2] +
              b[2] * rot_abs[1][1])
    return false;

  // A1 x B1
  s = t_vec[0] * rot[2][1] - t_vec[2] * rot[0][1];
  t = std::abs(s);
  if (t > a[0] * rot_abs[2][1] + a[2] * rot_abs[0][1] + b[0] * rot_abs[1][2] +
              b[2] * rot_abs[1][0])
    return false;

  // A1 x B2
  s = t_vec[0] * rot[2][2] - t_vec[2] * rot[0][2];
  t = std::abs(s);
  if (t > a[0] * rot_abs[2][2] + a[2] * rot_abs[0][2] + b[0] * rot_abs[1][1] +
              b[1] * rot_abs[1][0])
    return false;

  // A2 x B0
  s = t_vec[1] * rot[0][0] - t_vec[0] * rot[1][0];
  t = std::abs(s);
  if (t > a[0] * rot_abs[1][0] + a[1] * rot_abs[0][0] + b[1] * rot_abs[2][2] +
              b[2] * rot_abs[2][1])
    return false;

  // A2 x B1
  s = t_vec[1] * rot[0][1] - t_vec[0] * rot[1][1];
  t = std::abs(s);
  if (t > a[0] * rot_abs[1][1] + a[1] * rot_abs[0][1] + b[0] * rot_abs[2][2] +
              b[2] * rot_abs[2][0])
    return false;

  // A2 x B2
  s = t_vec[1] * rot[0][2] - t_vec[0] * rot[1][2];
  t = std::abs(s);
  if (t > a[0] * rot_abs[1][2] + a[1] * rot_abs[0][2] + b[0] * rot_abs[2][1] +
              b[1] * rot_abs[2][0])
    return false;

  return true;
}

} // namespace detail

template <std::size_t Dims, typename Policy0, typename Policy1>
auto obb_intersects_obb(const tf::obb_like<Dims, Policy0> &obb0,
                        const tf::obb_like<Dims, Policy1> &obb1) -> bool {
  static_assert(Dims == 2 || Dims == 3,
                "OBB intersection is implemented for 2D and 3D only");

  using T = tf::coordinate_type<Policy0, Policy1>;

  if constexpr (Dims == 2) {
    return detail::obb_intersects_obb_2d<T>(obb0, obb1);
  } else {
    return detail::obb_intersects_obb_3d<T>(obb0, obb1);
  }
}

} // namespace tf::core

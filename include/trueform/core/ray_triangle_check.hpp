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
#include "./epsilon.hpp"
#include "./polygon.hpp"
#include "./ray_cast_info.hpp"
#include "./ray_config.hpp"
#include "./ray_like.hpp"
#include <limits>

namespace tf::core {

/// @brief Moller-Trumbore ray-triangle intersection test.
///
/// Two cross products, three dots, early-out on barycentric coordinates.
/// Replaces the generic plane + containment path for triangles.
template <typename Policy0, typename Policy1>
auto ray_triangle_check(
    const tf::ray_like<3, Policy0> &ray,
    const tf::polygon<3, Policy1> &tri,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  constexpr auto eps = tf::epsilon<RealT>;
  constexpr auto meps = std::numeric_limits<RealT>::epsilon();

  auto e1 = tri[1] - tri[0];
  auto e2 = tri[2] - tri[0];
  auto s = ray.origin - tri[0];
  auto h = tf::cross(ray.direction, e2);
  auto det = tf::dot(e1, h);

  auto n = tf::cross(e1, e2);
  auto n_len2 = n.length2();
  if (det * det < ray.direction.length2() * n_len2 * meps) {
    auto d = tf::dot(n, s);
    if (d * d < n_len2 * meps * meps)
      return tf::make_ray_cast_info(tf::intersect_status::coplanar, RealT{0});
    return tf::make_ray_cast_info(tf::intersect_status::parallel, RealT{0});
  }

  auto inv_det = RealT{1} / det;
  auto u = inv_det * tf::dot(s, h);

  if (u < -eps || u > RealT{1} + eps)
    return tf::make_ray_cast_info(tf::intersect_status::none, RealT{0});

  auto q = tf::cross(s, e1);
  auto v = inv_det * tf::dot(ray.direction, q);

  if (v < -eps || u + v > RealT{1} + eps)
    return tf::make_ray_cast_info(tf::intersect_status::none, RealT{0});

  auto t = inv_det * tf::dot(e2, q);

  return tf::make_ray_cast_info(
      static_cast<tf::intersect_status>(
          char(t >= config.min_t - eps) &
          char(t <= config.max_t + eps)),
      t);
}

} // namespace tf::core

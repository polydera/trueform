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
#include "./aabb_like.hpp"
#include "./contains_coplanar_point.hpp"
#include "./dot.hpp"
#include "./epsilon.hpp"
#include "./epsilon_inverse.hpp"
#include "./line_like.hpp"
#include "./line_line_check.hpp"
#include "./parallelogram_area.hpp"
#include "./plane_like.hpp"
#include "./policy/plane.hpp"
#include "./polygon.hpp"
#include "./ray.hpp"
#include "./ray_aabb_check.hpp"
#include "./ray_cast_2d.hpp"
#include "./ray_cast_info.hpp"
#include "./ray_config.hpp"
#include "./ray_obb_check.hpp"
#include "./segment.hpp"

namespace tf {

/// @ingroup core_queries
/// @brief Cast a ray against a geometric primitive.
///
/// Returns a @ref tf::ray_cast_info containing the intersection status and
/// the parameter `t` at the hit point (if any). The hit point can be computed
/// as `ray.origin + t * ray.direction`.
///
/// @tparam Dims The dimensionality.
/// @tparam Policy0 The ray's storage policy.
/// @tparam Policy1 The target's storage policy.
/// @param ray The ray to cast.
/// @param plane The plane to test against.
/// @param config Optional ray configuration (min/max t values).
/// @return A @ref tf::ray_cast_info with intersection status and t parameter.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray,
    const tf::plane_like<Dims, Policy1> &plane,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto Vd = tf::dot(plane.normal, ray.direction);
  auto V0 = tf::dot(plane.normal, ray.origin) + plane.d;
  RealT t = 0;
  if (Vd * Vd <
      ray.direction.length2() * std::numeric_limits<RealT>::epsilon()) {
    if (std::abs(V0) < std::numeric_limits<RealT>::epsilon()) {
      return tf::make_ray_cast_info(tf::intersect_status::coplanar, t);
    } else {
      return tf::make_ray_cast_info(tf::intersect_status::parallel, t);
    }
  }

  t = -V0 / Vd;
  return tf::make_ray_cast_info(
      static_cast<tf::intersect_status>(
          char(t >= config.min_t - tf::epsilon<RealT>) &
          char(t <= config.max_t + tf::epsilon<RealT>)),
      t);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray,
    const tf::polygon<Dims, Policy1> &poly_in,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  if constexpr (Dims == 2) {
    return core::ray_cast_2d(ray, poly_in, config);
  } else {
    using RealT = tf::coordinate_type<Policy0, Policy1>;
    const auto &poly = tf::tag_plane(poly_in);
    auto result = ray_cast(ray, poly.plane(), config);
    if (result) {
      auto pt = ray.origin + result.t * ray.direction;
      result.status =
          static_cast<tf::intersect_status>(tf::contains_coplanar_point(
              poly, pt, tf::make_simple_projector(poly.normal()),
              tf::epsilon<RealT>));
    }
    return result;
  }
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray, const tf::segment<Dims, Policy1> &seg,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  if constexpr (Dims == 2) {
    return core::ray_cast_2d(ray, seg, config);
  } else {
    using RealT = tf::coordinate_type<Policy0, Policy1>;
    auto ray1 = tf::make_ray_between_points(seg[0], seg[1]);
    auto [status, t0, t1] = tf::core::line_line_check_full(ray, ray1);
    if (status == tf::intersect_status::non_parallel &&
        t0 >= config.min_t - tf::epsilon<RealT> &&
        t0 <= config.max_t + tf::epsilon<RealT> && t1 >= -tf::epsilon<RealT> &&
        t1 <= 1 + tf::epsilon<RealT>) {
      auto pt0 = ray.origin + t0 * ray.direction;
      auto pt1 = ray1.origin + t1 * ray1.direction;
      auto d2 = (pt0 - pt1).length2();
      status = static_cast<intersect_status>(d2 < tf::epsilon2<decltype(d2)>);
    }
    return tf::make_ray_cast_info(status, t0);
  }
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray,
    const tf::line_like<Dims, Policy1> &line,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  if constexpr (Dims == 2) {
    return core::ray_cast_2d(ray, line, config);
  } else {
    using RealT = tf::coordinate_type<Policy0, Policy1>;
    auto [status, t0, t1] = tf::core::line_line_check_full(ray, line);
    if (status == tf::intersect_status::non_parallel &&
        t0 >= config.min_t - tf::epsilon<RealT> &&
        t0 <= config.max_t + tf::epsilon<RealT>) {
      auto pt0 = ray.origin + t0 * ray.direction;
      auto pt1 = line.origin + t1 * line.direction;
      auto d2 = (pt0 - pt1).length2();
      status = static_cast<intersect_status>(d2 < tf::epsilon2<decltype(d2)>);
    }
    return tf::make_ray_cast_info(status, t0);
  }
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray, const tf::ray_like<Dims, Policy1> &ray1,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  if constexpr (Dims == 2) {
    return core::ray_cast_2d(ray, ray1, config);
  } else {
    using RealT = tf::coordinate_type<Policy0, Policy1>;
    auto [status, t0, t1] = tf::core::line_line_check_full(ray, ray1);
    if (status == tf::intersect_status::non_parallel &&
        t0 >= config.min_t - tf::epsilon<RealT> &&
        t0 <= config.max_t + tf::epsilon<RealT> && t1 >= -tf::epsilon<RealT>) {
      auto pt0 = ray.origin + t0 * ray.direction;
      auto pt1 = ray1.origin + t1 * ray1.direction;
      auto d2 = (pt0 - pt1).length2();
      status = static_cast<intersect_status>(d2 < tf::epsilon2<decltype(d2)>);
    }
    return tf::make_ray_cast_info(status, t0);
  }
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray,
    const tf::point_like<Dims, Policy1> &point,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto dist_vec = point - ray.origin;
  auto t = tf::dot(dist_vec, ray.direction) / ray.direction.length2();
  auto area2 = tf::parallelogram_area2(ray.direction, dist_vec);
  return tf::make_ray_cast_info(
      static_cast<tf::intersect_status>(
          char(area2 < tf::epsilon2<decltype(area2)>) &
          char(t >= config.min_t - tf::epsilon<RealT>) &
          char(t <= config.max_t + tf::epsilon<RealT>)),
      t);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray,
    const tf::aabb_like<Dims, Policy1> &aabb,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  tf::coordinate_type<Policy0, Policy1> t_min{}, t_max{};
  tf::vector<tf::coordinate_type<Policy0, Policy1>, Dims> ray_inv_dir;
  for (std::size_t i = 0; i < Dims; ++i)
    ray_inv_dir[i] = tf::epsilon_inverse(ray.direction[i]);
  auto status = core::ray_aabb_check(ray, ray_inv_dir, aabb, t_min, t_max,
                                     config.min_t, config.max_t);
  return tf::make_ray_cast_info(status, t_min);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray, const tf::obb_like<Dims, Policy1> &obb,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  tf::coordinate_type<Policy0, Policy1> t_min{}, t_max{};
  auto status =
      core::ray_obb_check(ray, obb, t_min, t_max, config.min_t, config.max_t);
  return tf::make_ray_cast_info(status, t_min);
}

} // namespace tf

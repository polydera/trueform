/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./aabb_like.hpp"
#include "./contains_coplanar_point.hpp"
#include "./dot.hpp"
#include "./epsilon_inverse.hpp"
#include "./line_like.hpp"
#include "./line_line_check.hpp"
#include "./parallelogram_area.hpp"
#include "./plane_like.hpp"
#include "./policy/plane.hpp"
#include "./polygon.hpp"
#include "./ray.hpp"
#include "./ray_aabb_check.hpp"
#include "./ray_cast_info.hpp"
#include "./ray_config.hpp"
#include "./segment.hpp"
#include <limits>

namespace tf {

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray,
    const tf::plane_like<Dims, Policy1> &plane,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto Vd = tf::dot(plane.normal, ray.direction);
  auto V0 = tf::dot(plane.normal, ray.origin) + plane.d;
  RealT t = 0;
  if (std::abs(Vd) < std::numeric_limits<RealT>::epsilon()) {
    if (std::abs(V0) < std::numeric_limits<RealT>::epsilon()) {
      return tf::make_ray_cast_info(tf::intersect_status::coplanar, t);
    } else {
      return tf::make_ray_cast_info(tf::intersect_status::parallel, t);
    }
  }

  t = -V0 / Vd;
  return tf::make_ray_cast_info(
      static_cast<tf::intersect_status>(
          char(t >= config.min_t - std::numeric_limits<RealT>::epsilon()) &
          char(t <= config.max_t + std::numeric_limits<RealT>::epsilon())),
      t);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray,
    const tf::polygon<Dims, Policy1> &poly_in,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  const auto &poly = tf::tag_plane(poly_in);
  auto result = ray_cast(ray, poly.plane(), config);
  if (result) {
    auto pt = ray.origin + result.t * ray.direction;
    result.status =
        static_cast<tf::intersect_status>(tf::contains_coplanar_point(
            poly, pt, tf::make_simple_projector(poly.normal()),
            std::numeric_limits<RealT>::epsilon()));
  }
  return result;
}

template <typename Policy0, typename Policy1>
auto ray_cast(
    const tf::ray_like<2, Policy0> &ray, const tf::polygon<2, Policy1> &poly,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;

  const std::size_t n = poly.size();
  std::size_t prev = n - 1;
  RealT closest_t = config.max_t + RealT(1);
  bool hit = false;

  for (std::size_t i = 0; i < n; prev = i++) {
    auto seg = tf::make_segment_between_points(poly[prev], poly[i]);
    auto info = tf::ray_cast(ray, seg, config);
    if (info && info.t < closest_t) {
      closest_t = info.t;
      hit = true;
    }
  }
  if (!hit && contains_coplanar_point(poly, ray.origin)) {
    hit = true;
    closest_t = 0;
  }

  return tf::make_ray_cast_info(static_cast<tf::intersect_status>(hit),
                                closest_t);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray, const tf::segment<Dims, Policy1> &seg,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto ray1 = tf::make_ray_between_points(seg[0], seg[1]);
  auto [status, t0, t1] = tf::core::line_line_check_full(ray, ray1);
  if (status == tf::intersect_status::non_parallel &&
      t0 >= config.min_t - std::numeric_limits<RealT>::epsilon() &&
      t0 <= config.max_t + std::numeric_limits<RealT>::epsilon() &&
      t1 >= -std::numeric_limits<RealT>::epsilon() &&
      t1 <= 1 + std::numeric_limits<RealT>::epsilon()) {
    auto pt0 = ray.origin + t0 * ray.direction;
    auto pt1 = ray1.origin + t1 * ray1.direction;
    auto d2 = (pt0 - pt1).length2();
    status = static_cast<intersect_status>(
        d2 < std::numeric_limits<decltype(d2)>::epsilon());
  }
  return tf::make_ray_cast_info(status, t0);
}

template <typename Policy0, typename Policy1>
auto ray_cast(
    const tf::ray_like<2, Policy0> &ray, const tf::segment<2, Policy1> &seg,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {

  using RealT = tf::coordinate_type<Policy0, Policy1>;
  using vec2 = tf::vector<RealT, 2>;

  const vec2 p = ray.origin.as_vector_view();
  const vec2 r = ray.direction;
  const vec2 q = seg[0].as_vector_view();
  const vec2 s = seg[1] - seg[0];

  const RealT rxs = tf::signed_parallelogram_area(r, s);
  const RealT q_p_x_r = tf::signed_parallelogram_area(q - p, r);

  constexpr RealT eps = std::numeric_limits<RealT>::epsilon();

  if (std::abs(rxs) < eps) {
    if (std::abs(q_p_x_r) < eps)
      return tf::make_ray_cast_info(tf::intersect_status::colinear, RealT(0));
    return tf::make_ray_cast_info(tf::intersect_status::parallel, RealT(0));
  }
  const RealT t = tf::signed_parallelogram_area(q - p, s) / rxs;
  const RealT u = q_p_x_r / rxs;

  const bool in_bounds = char(t >= config.min_t - eps) &
                         char(t <= config.max_t + eps) & char(u >= -eps) &
                         char(u <= RealT(1) + eps);

  return tf::make_ray_cast_info(static_cast<tf::intersect_status>(in_bounds),
                                t);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray,
    const tf::line_like<Dims, Policy1> &line,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto [status, t0, t1] = tf::core::line_line_check_full(ray, line);
  if (status == tf::intersect_status::non_parallel &&
      t0 >= config.min_t - std::numeric_limits<RealT>::epsilon() &&
      t0 <= config.max_t + std::numeric_limits<RealT>::epsilon()) {
    auto pt0 = ray.origin + t0 * ray.direction;
    auto pt1 = line.origin + t1 * line.direction;
    auto d2 = (pt0 - pt1).length2();
    status = static_cast<intersect_status>(
        d2 < std::numeric_limits<decltype(d2)>::epsilon());
  }
  return tf::make_ray_cast_info(status, t0);
}

template <typename Policy0, typename Policy1>
auto ray_cast(
    const tf::ray_like<2, Policy0> &ray, const tf::line_like<2, Policy1> &line,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {

  using RealT = tf::coordinate_type<Policy0, Policy1>;
  using vec2 = tf::vector<RealT, 2>;

  const vec2 p = ray.origin.as_vector_view();
  const vec2 r = ray.direction;
  const vec2 q = line.origin.as_vector_view();
  const vec2 s = line.direction;

  const RealT rxs = tf::signed_parallelogram_area(r, s);
  const RealT q_p_x_r = tf::signed_parallelogram_area(q - p, r);

  constexpr RealT eps = std::numeric_limits<RealT>::epsilon();

  if (std::abs(rxs) < eps) {
    if (std::abs(q_p_x_r) < eps)
      return tf::make_ray_cast_info(tf::intersect_status::colinear, RealT(0));
    return tf::make_ray_cast_info(tf::intersect_status::parallel, RealT(0));
  }

  const RealT t = tf::signed_parallelogram_area(q - p, s) / rxs;

  const bool in_bounds =
      char(t >= config.min_t - eps) & char(t <= config.max_t + eps);

  return tf::make_ray_cast_info(static_cast<tf::intersect_status>(in_bounds),
                                t);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray, const tf::ray_like<Dims, Policy1> &ray1,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto [status, t0, t1] = tf::core::line_line_check_full(ray, ray1);
  if (status == tf::intersect_status::non_parallel &&
      t0 >= config.min_t - std::numeric_limits<RealT>::epsilon() &&
      t0 <= config.max_t + std::numeric_limits<RealT>::epsilon() &&
      t1 >= -std::numeric_limits<RealT>::epsilon()) {
    auto pt0 = ray.origin + t0 * ray.direction;
    auto pt1 = ray1.origin + t1 * ray1.direction;
    auto d2 = (pt0 - pt1).length2();
    status = static_cast<intersect_status>(
        d2 < std::numeric_limits<decltype(d2)>::epsilon());
  }
  return tf::make_ray_cast_info(status, t0);
}

template <typename Policy0, typename Policy1>
auto ray_cast(
    const tf::ray_like<2, Policy0> &ray, const tf::ray_like<2, Policy1> &ray1,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {

  using RealT = tf::coordinate_type<Policy0, Policy1>;
  using vec2 = tf::vector<RealT, 2>;

  const vec2 p = ray.origin.as_vector_view();
  const vec2 r = ray.direction;
  const vec2 q = ray1.origin.as_vector_view();
  const vec2 s = ray1.direction;

  const RealT rxs = tf::signed_parallelogram_area(r, s);
  const RealT q_p_x_r = tf::signed_parallelogram_area(q - p, r);

  constexpr RealT eps = std::numeric_limits<RealT>::epsilon();

  if (std::abs(rxs) < eps) {
    if (std::abs(q_p_x_r) < eps)
      return tf::make_ray_cast_info(tf::intersect_status::colinear, RealT(0));
    return tf::make_ray_cast_info(tf::intersect_status::parallel, RealT(0));
  }

  const RealT t = tf::signed_parallelogram_area(q - p, s) / rxs;
  const RealT u = q_p_x_r / rxs;

  const bool in_bounds = char(t >= config.min_t - eps) &
                         char(t <= config.max_t + eps) & char(u >= -eps);

  return tf::make_ray_cast_info(static_cast<tf::intersect_status>(in_bounds),
                                t);
}

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
          char(area2 < std::numeric_limits<decltype(area2)>::epsilon()) &
          char(t >= config.min_t - std::numeric_limits<RealT>::epsilon()) &
          char(t <= config.max_t + std::numeric_limits<RealT>::epsilon())),
      t);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray,
    const tf::aabb_like<Dims, Policy1> &aabb,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  tf::coordinate_type<Policy0, Policy1> t_min, t_max;
  tf::vector<tf::coordinate_type<Policy0, Policy1>, Dims> ray_inv_dir;
  for (std::size_t i = 0; i < Dims; ++i)
    ray_inv_dir[i] = tf::epsilon_inverse(ray.direction[i]);
  auto status = core::ray_aabb_check(ray, ray_inv_dir, aabb, t_min, t_max,
                                     config.min_t, config.max_t);
  return tf::make_ray_cast_info(status, t_min);
}
} // namespace tf

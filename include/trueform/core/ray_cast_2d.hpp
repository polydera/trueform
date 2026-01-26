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
#include "./contains_coplanar_point.hpp"
#include "./ray_like.hpp"
#include "./line_like.hpp"
#include "./parallelogram_area.hpp"
#include "./polygon.hpp"
#include "./ray_cast_info.hpp"
#include "./ray_config.hpp"
#include "./segment.hpp"

namespace tf::core {

/// @brief 2D ray-segment intersection
template <typename Policy0, typename Policy1>
auto ray_cast_2d(
    const tf::ray_like<2, Policy0> &ray, const tf::segment<2, Policy1> &seg,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config) {

  using RealT = tf::coordinate_type<Policy0, Policy1>;
  using vec2 = tf::vector<RealT, 2>;

  const vec2 p = ray.origin.as_vector_view();
  const vec2 r = ray.direction;
  const vec2 q = seg[0].as_vector_view();
  const vec2 s = seg[1] - seg[0];

  const RealT rxs = tf::signed_parallelogram_area(r, s);
  const RealT q_p_x_r = tf::signed_parallelogram_area(q - p, r);

  constexpr RealT eps = tf::epsilon<RealT>;

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

/// @brief 2D ray-line intersection
template <typename Policy0, typename Policy1>
auto ray_cast_2d(
    const tf::ray_like<2, Policy0> &ray, const tf::line_like<2, Policy1> &line,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config) {

  using RealT = tf::coordinate_type<Policy0, Policy1>;
  using vec2 = tf::vector<RealT, 2>;

  const vec2 p = ray.origin.as_vector_view();
  const vec2 r = ray.direction;
  const vec2 q = line.origin.as_vector_view();
  const vec2 s = line.direction;

  const RealT rxs = tf::signed_parallelogram_area(r, s);
  const RealT q_p_x_r = tf::signed_parallelogram_area(q - p, r);

  constexpr RealT eps = tf::epsilon<RealT>;

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

/// @brief 2D ray-ray intersection
template <typename Policy0, typename Policy1>
auto ray_cast_2d(
    const tf::ray_like<2, Policy0> &ray, const tf::ray_like<2, Policy1> &ray1,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config) {

  using RealT = tf::coordinate_type<Policy0, Policy1>;
  using vec2 = tf::vector<RealT, 2>;

  const vec2 p = ray.origin.as_vector_view();
  const vec2 r = ray.direction;
  const vec2 q = ray1.origin.as_vector_view();
  const vec2 s = ray1.direction;

  const RealT rxs = tf::signed_parallelogram_area(r, s);
  const RealT q_p_x_r = tf::signed_parallelogram_area(q - p, r);

  constexpr RealT eps = tf::epsilon<RealT>;

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

/// @brief 2D ray-polygon intersection (boundary test)
template <typename Policy0, typename Policy1>
auto ray_cast_2d(
    const tf::ray_like<2, Policy0> &ray, const tf::polygon<2, Policy1> &poly,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;

  const std::size_t n = poly.size();
  std::size_t prev = n - 1;
  RealT closest_t = config.max_t + RealT(1);
  bool hit = false;

  for (std::size_t i = 0; i < n; prev = i++) {
    auto seg = tf::make_segment_between_points(poly[prev], poly[i]);
    auto info = ray_cast_2d(ray, seg, config);
    if (info && info.t < closest_t) {
      closest_t = info.t;
      hit = true;
    }
  }
  if (!hit && contains_coplanar_point(poly, ray(config.min_t))) {
    hit = true;
    closest_t = config.min_t;
  }

  return tf::make_ray_cast_info(static_cast<tf::intersect_status>(hit),
                                closest_t);
}

} // namespace tf::core

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
#include "./line_like.hpp"
#include "./polygon.hpp"
#include "./ray_cast.hpp"
#include "./ray_hit_info.hpp"
#include "./segment.hpp"

namespace tf::core {

/// @brief 2D ray-polygon hit
template <typename Policy0, typename Policy1>
auto ray_hit_2d(
    const tf::ray_like<2, Policy0> &ray, const tf::polygon<2, Policy1> &poly,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto result = ray_cast(ray, poly, config);
  tf::ray_hit_info<RealT, 2> out;
  out.status = result.status;
  out.t = result.t;
  if (result)
    out.point = ray.origin + result.t * ray.direction;
  return out;
}

/// @brief 2D ray-segment hit
template <typename Policy0, typename Policy1>
auto ray_hit_2d(
    const tf::ray_like<2, Policy0> &ray, const tf::segment<2, Policy1> &seg,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto result = ray_cast(ray, seg, config);
  tf::ray_hit_info<RealT, 2> out;
  out.status = result.status;
  out.t = result.t;
  if (result)
    out.point = ray.origin + result.t * ray.direction;
  return out;
}

/// @brief 2D ray-line hit
template <typename Policy0, typename Policy1>
auto ray_hit_2d(
    const tf::ray_like<2, Policy0> &ray, const tf::line_like<2, Policy1> &line,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto result = ray_cast(ray, line, config);
  tf::ray_hit_info<RealT, 2> out;
  out.status = result.status;
  out.t = result.t;
  if (result)
    out.point = ray.origin + result.t * ray.direction;
  return out;
}

/// @brief 2D ray-ray hit
template <typename Policy0, typename Policy1>
auto ray_hit_2d(
    const tf::ray_like<2, Policy0> &ray, const tf::ray_like<2, Policy1> &ray1,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto result = ray_cast(ray, ray1, config);
  tf::ray_hit_info<RealT, 2> out;
  out.status = result.status;
  out.t = result.t;
  if (result)
    out.point = ray.origin + result.t * ray.direction;
  return out;
}

} // namespace tf::core

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./ray_cast.hpp"
#include "./ray_hit_info.hpp"

namespace tf {

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<Dims, Policy0> &ray,
    const tf::plane_like<Dims, Policy1> &plane,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto result = ray_cast(ray, plane, config);
  tf::ray_hit_info<RealT, Dims> out;
  out.status = result.status;
  out.t = result.t;
  if (result)
    out.point = ray.origin + result.t * ray.direction;
  return out;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<Dims, Policy0> &ray,
    const tf::polygon<Dims, Policy1> &poly_in,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;

  const auto &poly = tf::tag_plane(poly_in);
  auto result = ray_cast(ray, poly.plane(), config);
  tf::ray_hit_info<RealT, Dims> out;
  out.status = result.status;
  out.t = result.t;
  if (result) {
    out.point = ray.origin + result.t * ray.direction;
    out.status = static_cast<tf::intersect_status>(tf::contains_coplanar_point(
        poly, out.point, tf::make_simple_projector(poly.normal()),
        std::numeric_limits<RealT>::epsilon()));
  }
  return out;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<Dims, Policy0> &ray, const tf::segment<Dims, Policy1> &seg,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto ray1 = tf::make_ray_between_points(seg[0], seg[1]);
  auto [non_parallel, t0, t1] = tf::core::line_line_check(ray, ray1);
  intersect_status status = intersect_status::none;
  tf::point<tf::coordinate_type<decltype(t0), decltype(t1)>, Dims> pt{};
  if (non_parallel &&
      t0 >= config.min_t - std::numeric_limits<RealT>::epsilon() &&
      t0 <= config.max_t + std::numeric_limits<RealT>::epsilon() &&
      t1 >= -std::numeric_limits<RealT>::epsilon() &&
      t1 <= 1 + std::numeric_limits<RealT>::epsilon()) {
    auto pt0 = ray.origin + t0 * ray.direction;
    auto pt1 = ray1.origin + t1 * ray1.direction;
    auto d2 = (pt0 - pt1).length2();
    status = static_cast<intersect_status>(
        d2 < std::numeric_limits<decltype(d2)>::epsilon());
    auto pt_view = pt.as_vector_view();
    pt_view = (pt0.as_vector_view() + pt1.as_vector_view()) / 2;
  }
  return tf::make_ray_hit_info(status, t0, pt);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<Dims, Policy0> &ray,
    const tf::line_like<Dims, Policy1> &line,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto [non_parallel, t0, t1] = tf::core::line_line_check(ray, line);
  intersect_status status = intersect_status::none;
  tf::point<tf::coordinate_type<Policy0, Policy1>, Dims> pt{};
  if (non_parallel &&
      t0 >= config.min_t - std::numeric_limits<RealT>::epsilon() &&
      t0 <= config.max_t + std::numeric_limits<RealT>::epsilon()) {
    auto pt0 = ray.origin + t0 * ray.direction;
    auto pt1 = line.origin + t1 * line.direction;
    auto d2 = (pt0 - pt1).length2();
    status = static_cast<intersect_status>(
        d2 < std::numeric_limits<decltype(d2)>::epsilon());
    auto pt_view = pt.as_vector_view();
    pt_view = (pt0.as_vector_view() + pt1.as_vector_view()) / 2;
  }
  return tf::make_ray_hit_info(status, t0, pt);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<Dims, Policy0> &ray, const tf::ray_like<Dims, Policy1> &ray1,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto [non_parallel, t0, t1] = tf::core::line_line_check(ray, ray1);
  intersect_status status = intersect_status::none;
  tf::point<tf::coordinate_type<Policy0, Policy1>, Dims> pt{};
  if (non_parallel &&
      t0 >= config.min_t - std::numeric_limits<RealT>::epsilon() &&
      t0 <= config.max_t + std::numeric_limits<RealT>::epsilon() &&
      t1 >= -std::numeric_limits<RealT>::epsilon()) {
    auto pt0 = ray.origin + t0 * ray.direction;
    auto pt1 = ray1.origin + t1 * ray1.direction;
    auto d2 = (pt0 - pt1).length2();
    status = static_cast<intersect_status>(
        d2 < std::numeric_limits<decltype(d2)>::epsilon());
    auto pt_view = pt.as_vector_view();
    pt_view = (pt0.as_vector_view() + pt1.as_vector_view()) / 2;
  }
  return tf::make_ray_hit_info(status, t0, pt);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<Dims, Policy0> &ray,
    const tf::point_like<Dims, Policy1> &point,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  auto result = ray_cast(ray, point, config);
  tf::ray_hit_info<tf::coordinate_type<Policy0, Policy1>, Dims> out;
  out.status = result.status;
  out.t = result.t;
  if (result) {
    out.point = ray.origin + result.t * ray.direction;
  }
  return out;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<Dims, Policy0> &ray,
    const tf::aabb_like<Dims, Policy1> &aabb,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  auto result = ray_cast(ray, aabb, config);
  tf::ray_hit_info<tf::coordinate_type<Policy0, Policy1>, Dims> out;
  out.status = result.status;
  out.t = result.t;
  if (result) {
    out.point = ray.origin + result.t * ray.direction;
  }
  return out;
}
} // namespace tf

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
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
        tf::epsilon<RealT>));
  }
  return out;
}

template <typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<2, Policy0> &ray, const tf::polygon<2, Policy1> &poly,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto result = ray_cast(ray, poly, config);
  tf::ray_hit_info<RealT, 2> out;
  out.status = result.status;
  out.t = result.t;
  if (result)
    out.point = ray.origin + result.t * ray.direction;
  return out;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<Dims, Policy0> &ray, const tf::segment<Dims, Policy1> &seg,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto ray1 = tf::make_ray_between_points(seg[0], seg[1]);
  auto [status, t0, t1] = tf::core::line_line_check_full(ray, ray1);
  tf::point<tf::coordinate_type<decltype(t0), decltype(t1)>, Dims> pt{};
  if (status == tf::intersect_status::non_parallel &&
      t0 >= config.min_t - tf::epsilon<RealT> &&
      t0 <= config.max_t + tf::epsilon<RealT> &&
      t1 >= -tf::epsilon<RealT> &&
      t1 <= 1 + tf::epsilon<RealT>) {
    auto pt0 = ray.origin + t0 * ray.direction;
    auto pt1 = ray1.origin + t1 * ray1.direction;
    auto d2 = (pt0 - pt1).length2();
    status = static_cast<intersect_status>(
        d2 < tf::epsilon2<decltype(d2)>);
    auto pt_view = pt.as_vector_view();
    pt_view = (pt0.as_vector_view() + pt1.as_vector_view()) / 2;
  }
  return tf::make_ray_hit_info(status, t0, pt);
}

template <typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<2, Policy0> &ray, const tf::segment<2, Policy1> &seg,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto result = ray_cast(ray, seg, config);
  tf::ray_hit_info<RealT, 2> out;
  out.status = result.status;
  out.t = result.t;
  if (result)
    out.point = ray.origin + result.t * ray.direction;
  return out;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<Dims, Policy0> &ray,
    const tf::line_like<Dims, Policy1> &line,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto [status, t0, t1] = tf::core::line_line_check_full(ray, line);
  tf::point<tf::coordinate_type<Policy0, Policy1>, Dims> pt{};
  if (status == tf::intersect_status::non_parallel &&
      t0 >= config.min_t - tf::epsilon<RealT> &&
      t0 <= config.max_t + tf::epsilon<RealT>) {
    auto pt0 = ray.origin + t0 * ray.direction;
    auto pt1 = line.origin + t1 * line.direction;
    auto d2 = (pt0 - pt1).length2();
    status = static_cast<intersect_status>(
        d2 < tf::epsilon2<decltype(d2)>);
    auto pt_view = pt.as_vector_view();
    pt_view = (pt0.as_vector_view() + pt1.as_vector_view()) / 2;
  }
  return tf::make_ray_hit_info(status, t0, pt);
}

template <typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<2, Policy0> &ray, const tf::line_like<2, Policy1> &line,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto result = ray_cast(ray, line, config);
  tf::ray_hit_info<RealT, 2> out;
  out.status = result.status;
  out.t = result.t;
  if (result)
    out.point = ray.origin + result.t * ray.direction;
  return out;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<Dims, Policy0> &ray, const tf::ray_like<Dims, Policy1> &ray1,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto [status, t0, t1] = tf::core::line_line_check_full(ray, ray1);
  tf::point<tf::coordinate_type<Policy0, Policy1>, Dims> pt{};
  if (status == tf::intersect_status::non_parallel &&
      t0 >= config.min_t - tf::epsilon<RealT> &&
      t0 <= config.max_t + tf::epsilon<RealT> &&
      t1 >= -tf::epsilon<RealT>) {
    auto pt0 = ray.origin + t0 * ray.direction;
    auto pt1 = ray1.origin + t1 * ray1.direction;
    auto d2 = (pt0 - pt1).length2();
    status = static_cast<intersect_status>(
        d2 < tf::epsilon2<decltype(d2)>);
    auto pt_view = pt.as_vector_view();
    pt_view = (pt0.as_vector_view() + pt1.as_vector_view()) / 2;
  }
  return tf::make_ray_hit_info(status, t0, pt);
}

template <typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<2, Policy0> &ray, const tf::ray_like<2, Policy1> &ray1,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto result = ray_cast(ray, ray1, config);
  tf::ray_hit_info<RealT, 2> out;
  out.status = result.status;
  out.t = result.t;
  if (result)
    out.point = ray.origin + result.t * ray.direction;
  return out;
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

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<Dims, Policy0> &ray,
    const tf::obb_like<Dims, Policy1> &obb,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config = {}) {
  auto result = ray_cast(ray, obb, config);
  tf::ray_hit_info<tf::coordinate_type<Policy0, Policy1>, Dims> out;
  out.status = result.status;
  out.t = result.t;
  if (result) {
    out.point = ray.origin + result.t * ray.direction;
  }
  return out;
}

} // namespace tf

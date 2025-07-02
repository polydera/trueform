/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./aabb_like.hpp"
#include "./closest_point_parametric.hpp"
#include "./line.hpp"
#include "./point_like.hpp"
#include "./polygon.hpp"
#include "./ray.hpp"
#include "./ray_cast.hpp"
#include "./segment.hpp"

namespace tf {

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between
/// two AABBs.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const aabb_like<Dims, Policy0> &a,
                const aabb_like<Dims, Policy1> &b) -> bool {
  for (std::size_t i = 0; i < Dims; ++i) {
    if (a.max[i] < b.min[i] || b.max[i] < a.min[i])
      return false;
  }
  return true;
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect within epsilon.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const aabb_like<Dims, Policy0> &a,
                const aabb_like<Dims, Policy1> &b,
                tf::coordinate_type<Policy0, Policy1> epsilon) -> bool {
  for (std::size_t i = 0; i < Dims; ++i) {
    if (a.max[i] + epsilon < b.min[i] || b.max[i] + epsilon < a.min[i])
      return false;
  }
  return true;
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.

template <std::size_t N, typename T0, typename T1>
auto intersects(const point_like<N, T0> &point, const aabb_like<N, T1> &box)
    -> bool {
  for (std::size_t i = 0; i < N; ++i) {
    if (point[i] < box.min[i] || point[i] > box.max[i])
      return false;
  }
  return true;
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect within epsilon.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.

template <std::size_t N, typename Policy, typename T1>
auto intersects(const aabb_like<N, Policy> &box, const point_like<N, T1> &point)
    -> bool {
  return intersects(point, box);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect within epsilon.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.

template <std::size_t N, typename T0, typename T1>
auto intersects(const point_like<N, T0> &point, const aabb_like<N, T1> &box,
                tf::coordinate_type<T0, T1> epsilon) -> bool {
  for (std::size_t i = 0; i < N; ++i) {
    if (point[i] + epsilon < box.min[i] || point[i] - epsilon > box.max[i])
      return false;
  }
  return true;
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect within epsilon.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t N, typename Policy, typename T1>
auto intersects(const aabb_like<N, Policy> &box, const point_like<N, T1> &point,
                tf::coordinate_type<Policy, T1> epsilon) -> bool {
  return intersects(point, box, epsilon);
}
/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect within epsilon.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t N, typename T0, typename T1>
auto intersects(const point_like<N, T0> &v0, const point_like<N, T1> &v1,
                tf::coordinate_type<T0, T1> epsilon) -> bool {
  return (v0 - v1).length2() < epsilon * epsilon;
}
/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t N, typename T0, typename T1>
auto intersects(const tf::point_like<N, T0> &v0,
                const tf::point_like<N, T1> &v1) -> bool {
  return (v0 - v1).length2() <
         std::numeric_limits<tf::coordinate_type<T0, T1>>::epsilon();
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy, typename T1>
auto intersects(const tf::line_like<Dims, Policy> &l,
                const tf::point_like<Dims, T1> &v1) {
  auto t = tf::closest_point_parametric(l, v1);
  auto pt = l.origin + t * l.direction;
  auto d2 = (pt - v1).length2();
  return d2 < std::numeric_limits<decltype(d2)>::epsilon();
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename T1, typename Policy>
auto intersects(const tf::point_like<Dims, T1> &v0,
                const tf::line_like<Dims, Policy> &l) {
  return intersects(l, v0);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy, typename T1>
auto intersects(const tf::ray_like<Dims, Policy> &r,
                const tf::point_like<Dims, T1> &v1) {
  auto t = tf::closest_point_parametric(r, v1);
  auto pt = r.origin + t * r.direction;
  auto d2 = (pt - v1).length2();
  return d2 < std::numeric_limits<decltype(d2)>::epsilon();
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename T1, typename Policy>
auto intersects(const tf::point_like<Dims, T1> &v0,
                const tf::ray_like<Dims, Policy> &r) {
  return intersects(r, v0);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <typename T0, std::size_t Dims, typename T1>
auto intersects(const tf::segment<Dims, T0> &s,
                const tf::point_like<Dims, T1> &v1) {
  auto t = tf::closest_point_parametric(s, v1);
  auto l = tf::make_line_between_points(s[0], s[1]);
  auto pt = l.origin + t * l.direction;
  auto d2 = (pt - v1).length2();
  return d2 < std::numeric_limits<decltype(d2)>::epsilon();
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename T0, typename T1>
auto intersects(const tf::point_like<Dims, T0> &v0,
                const tf::segment<Dims, T1> &s) {
  return intersects(s, v0);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::line_like<Dims, Policy0> &l0,
                const tf::line_like<Dims, Policy1> &l1) {
  auto [t0, t1] = tf::closest_point_parametric(l0, l1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  auto d2 = (pt0 - pt1).length2();
  return d2 < std::numeric_limits<decltype(d2)>::epsilon();
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::ray_like<Dims, Policy0> &r0,
                const tf::ray_like<Dims, Policy1> &r1) {
  auto [t0, t1] = tf::closest_point_parametric(r0, r1);
  auto pt0 = r0.origin + t0 * r0.direction;
  auto pt1 = r1.origin + t1 * r1.direction;
  auto d2 = (pt0 - pt1).length2();
  return d2 < std::numeric_limits<decltype(d2)>::epsilon();
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::line_like<Dims, Policy0> &l0,
                const tf::ray_like<Dims, Policy1> &r1) {
  auto [t0, t1] = tf::closest_point_parametric(l0, r1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = r1.origin + t1 * r1.direction;
  auto d2 = (pt0 - pt1).length2();
  return d2 < std::numeric_limits<decltype(d2)>::epsilon();
}
/// @ingroup geometry
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::ray_like<Dims, Policy0> &r0,
                const tf::line_like<Dims, Policy1> &l1) {
  return intersects(l1, r0);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy, typename T>
auto intersects(const tf::ray_like<Dims, Policy> &r0,
                const tf::segment<Dims, T> &s1) {
  auto l1 = tf::make_line_between_points(s1[0], s1[1]);
  auto [t0, t1] = tf::closest_point_parametric(r0, l1);
  auto pt0 = r0.origin + t0 * r0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  auto d2 = (pt0 - pt1).length2();
  return d2 < std::numeric_limits<decltype(d2)>::epsilon();
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy, typename T>
auto intersects(const tf::line_like<Dims, Policy> &l0,
                const tf::segment<Dims, T> &s1) {
  auto l1 = tf::make_line_between_points(s1[0], s1[1]);
  auto [t0, t1] = tf::closest_point_parametric(l0, l1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  auto d2 = (pt0 - pt1).length2();
  return d2 < std::numeric_limits<decltype(d2)>::epsilon();
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <typename T, std::size_t Dims, typename Policy>
auto intersects(const tf::segment<Dims, T> &s0,
                const tf::line_like<Dims, Policy> &l1) {
  auto l0 = tf::make_line_between_points(s0[0], s0[1]);
  auto [t0, t1] = tf::closest_point_parametric(l0, l1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  auto d2 = (pt0 - pt1).length2();
  return d2 < std::numeric_limits<decltype(d2)>::epsilon();
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <typename T, std::size_t Dims, typename Policy>
auto intersects(const tf::segment<Dims, T> &s0,
                const tf::ray_like<Dims, Policy> &r1) {
  auto l0 = tf::make_line_between_points(s0[0], s0[1]);
  auto [t0, t1] = tf::closest_point_parametric(l0, r1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = r1.origin + t1 * r1.direction;
  auto d2 = (pt0 - pt1).length2();
  return d2 < std::numeric_limits<decltype(d2)>::epsilon();
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename T0, typename T1>
auto intersects(const tf::segment<Dims, T0> &s0,
                const tf::segment<Dims, T1> &s1) {
  auto l0 = tf::make_line_between_points(s0[0], s0[1]);
  auto l1 = tf::make_line_between_points(s1[0], s1[1]);
  auto [t0, t1] = tf::closest_point_parametric(l0, l1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  auto d2 = (pt0 - pt1).length2();
  return d2 < std::numeric_limits<decltype(d2)>::epsilon();
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <typename Policy0, std::size_t Dims, typename Policy1>
auto intersects(const tf::polygon<Dims, Policy0> &poly_in,
                const tf::point_like<Dims, Policy1> &pt) -> bool {
  const auto &poly = tf::tag_plane(poly_in);
  auto d = tf::dot(poly.plane().normal, pt) + poly.plane().d;
  auto c_pt = pt - d * poly.plane().normal;
  return std::abs(d) < std::numeric_limits<decltype(d)>::epsilon() &&
         tf::contains_coplanar_point(poly, c_pt);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::point_like<Dims, Policy0> &pt,
                const tf::polygon<Dims, Policy1> &poly) -> bool {
  return intersects(poly, pt);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <typename Policy0, std::size_t Dims, typename Policy>
auto intersects(const tf::polygon<Dims, Policy0> &poly_in,
                const tf::ray_like<Dims, Policy> &ray) -> bool {
  const auto &poly = tf::tag_plane(poly_in);
  return tf::ray_cast(ray, poly);
  ;
}
/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy, typename Policy0>
auto intersects(const tf::ray_like<Dims, Policy> &ray,
                const tf::polygon<Dims, Policy0> &poly) {
  return intersects(poly, ray);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <typename Policy0, std::size_t Dims, typename Policy>
auto intersects(const tf::polygon<Dims, Policy0> &poly_in,
                const tf::line_like<Dims, Policy> &line) -> bool {
  using RealT = tf::coordinate_type<Policy0, Policy>;
  const auto &poly = tf::tag_plane(poly_in);
  return tf::ray_cast(tf::make_ray(line.origin, line.direction), poly,
                      tf::make_ray_config(-std::numeric_limits<RealT>::max(),
                                          std::numeric_limits<RealT>::max()));
  ;
}
/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy, typename Policy0>
auto intersects(const tf::line_like<Dims, Policy> &line,
                const tf::polygon<Dims, Policy0> &poly) {
  return intersects(poly, line);
}
/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::polygon<Dims, Policy0> &poly_in,
                const tf::segment<Dims, Policy1> &seg1) -> bool {
  const auto &poly = tf::tag_plane(poly_in);
  auto ray = tf::make_ray_between_points(seg1[0], seg1[1]);
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  return tf::ray_cast(ray, poly, tf::make_ray_config(RealT(0), RealT(1)));
}
/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <typename Policy, std::size_t Dims, typename Policy0>
auto intersects(const tf::segment<Dims, Policy> &seg,
                const tf::polygon<Dims, Policy0> &poly) {
  return intersects(poly, seg);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::polygon<Dims, Policy0> &poly_in0,
                const tf::polygon<Dims, Policy1> &poly_in1) -> bool {
  const auto &poly0 = tf::tag_plane(poly_in0);

  std::size_t size = poly0.size();
  std::size_t prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++) {
    if (intersects(poly0, tf::make_segment_between_points(poly_in1[prev],
                                                          poly_in1[i])))
      return true;
  }
  const auto &poly1 = tf::tag_plane(poly_in1);

  size = poly1.size();
  prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++) {
    if (intersects(poly1,
                   tf::make_segment_between_points(poly0[prev], poly0[i])))
      return true;
  }
  return false;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::plane_like<Dims, Policy0> &plane,
                const tf::point_like<Dims, Policy1> &pt) -> bool {
  auto d = tf::dot(plane.normal, pt) + plane.d;
  ;
  return d < std::numeric_limits<decltype(d)>::epsilon();
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::point_like<Dims, Policy0> &pt,
                const tf::plane_like<Dims, Policy0> &plane) -> bool {
  return intersects(plane, pt);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::plane_like<Dims, Policy0> &plane,
                const tf::ray_like<Dims, Policy1> &ray) -> bool {
  return tf::ray_cast(ray, plane);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::ray_like<Dims, Policy0> &ray,
                const tf::plane_like<Dims, Policy0> &plane) -> bool {
  return tf::ray_cast(ray, plane);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::plane_like<Dims, Policy0> &plane,
                const tf::line_like<Dims, Policy1> &line) -> bool {
  return tf::ray_cast(
      tf::make_ray_like(line.origin, line.direction), plane,
      tf::make_ray_config(
          std::numeric_limits<tf::coordinate_type<Policy0, Policy1>>::lowest(),
          std::numeric_limits<tf::coordinate_type<Policy0, Policy1>>::max()));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::line_like<Dims, Policy0> &line,
                const tf::plane_like<Dims, Policy0> &plane) -> bool {
  return intersects(plane, line);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::plane_like<Dims, Policy0> &plane,
                const tf::segment<Dims, Policy1> &seg) -> bool {
  return tf::ray_cast(
      tf::make_ray_like(seg[0], seg[1] - seg[0]), plane,
      tf::make_ray_config(tf::coordinate_type<Policy0, Policy1>(0),
                          tf::coordinate_type<Policy0, Policy1>(1)));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::segment<Dims, Policy0> &seg,
                const tf::plane_like<Dims, Policy0> &plane) -> bool {
  return intersects(plane, seg);
}

namespace core {
template <typename Obj> struct intersector_with {
  Obj obj;
  template <typename T> auto operator()(const T &t) const -> bool {
    return tf::intersects(obj, t);
  }
};

struct intersector {
  template <typename T> auto operator()(T &&t) const {
    return core::intersector_with<std::decay_t<T>>{static_cast<T &&>(t)};
  }

  template <typename T0, typename T1>
  auto operator()(const T0 &t0, const T1 &t1) const {
    return tf::intersects(t0, t1);
  }
};
} // namespace core

constexpr core::intersector intersects_f;
} // namespace tf

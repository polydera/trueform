/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./aabb.hpp"
#include "./closest_point_parametric.hpp"
#include "./form.hpp"
#include "./line.hpp"
#include "./point_like.hpp"
#include "./polygon.hpp"
#include "./ray.hpp"
#include "./ray_cast.hpp"
#include "./search.hpp"
#include "./segment.hpp"

namespace tf {

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between
/// two AABBs.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <typename T, std::size_t N>
auto intersects(const aabb<T, N> &a, const aabb<T, N> &b) -> bool {
  for (std::size_t i = 0; i < N; ++i) {
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

template <typename T, std::size_t N>
auto intersects(const aabb<T, N> &a, const aabb<T, N> &b, T epsilon) -> bool {
  for (std::size_t i = 0; i < N; ++i) {
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
auto intersects(const point_like<N, T0> &point, const aabb<T1, N> &box)
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

template <typename T, std::size_t N, typename T1>
auto intersects(const aabb<T, N> &box, const point_like<N, T1> &point) -> bool {
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
auto intersects(const point_like<N, T0> &point, const aabb<T1, N> &box,
                T1 epsilon) -> bool {
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
template <typename T, std::size_t N, typename T1>
auto intersects(const aabb<T, N> &box, const point_like<N, T1> &point,
                T epsilon) -> bool {
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
                tf::common_value<T0, T1> epsilon) -> bool {
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
         std::numeric_limits<tf::common_value<T0, T1>>::epsilon();
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <typename RealT, std::size_t Dims, typename T1>
auto intersects(const tf::line<RealT, Dims> &l,
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
template <std::size_t Dims, typename T1, typename RealT>
auto intersects(const tf::point_like<Dims, T1> &v0,
                const tf::line<RealT, Dims> &l) {
  return intersects(l, v0);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <typename RealT, std::size_t Dims, typename T1>
auto intersects(const tf::ray<RealT, Dims> &r,
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
template <std::size_t Dims, typename T1, typename RealT>
auto intersects(const tf::point_like<Dims, T1> &v0,
                const tf::ray<RealT, Dims> &r) {
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
auto intersects(const tf::segment<T0> &s, const tf::point_like<Dims, T1> &v1) {
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
auto intersects(const tf::point_like<Dims, T0> &v0, const tf::segment<T1> &s) {
  return intersects(s, v0);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <typename RealT, std::size_t Dims>
auto intersects(const tf::line<RealT, Dims> &l0,
                const tf::line<RealT, Dims> &l1) {
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
template <typename RealT, std::size_t Dims>
auto intersects(const tf::ray<RealT, Dims> &r0,
                const tf::ray<RealT, Dims> &r1) {
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
template <typename RealT, std::size_t Dims>
auto intersects(const tf::line<RealT, Dims> &l0,
                const tf::ray<RealT, Dims> &r1) {
  auto [t0, t1] = tf::closest_point_parametric(l0, r1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = r1.origin + t1 * r1.direction;
  auto d2 = (pt0 - pt1).length2();
  return d2 < std::numeric_limits<decltype(d2)>::epsilon();
}
/// @ingroup geometry
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <typename RealT, std::size_t Dims>
auto intersects(const tf::ray<RealT, Dims> &r0,
                const tf::line<RealT, Dims> &l1) {
  return intersects(l1, r0);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <typename RealT, std::size_t Dims, typename T>
auto intersects(const tf::ray<RealT, Dims> &r0, const tf::segment<T> &s1) {
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
template <typename RealT, std::size_t Dims, typename T>
auto intersects(const tf::line<RealT, Dims> &l0, const tf::segment<T> &s1) {
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
template <typename T, std::size_t Dims, typename RealT>
auto intersects(const tf::segment<T> &s0, const tf::line<RealT, Dims> &l1) {
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
template <typename T, std::size_t Dims, typename RealT>
auto intersects(const tf::segment<T> &s0, const tf::ray<RealT, Dims> &r1) {
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
template <typename T0, typename T1>
auto intersects(const tf::segment<T0> &s0, const tf::segment<T1> &s1) {
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
template <std::size_t V0, typename Policy0, std::size_t Dims, typename Policy1>
auto intersects(const tf::polygon<V0, Policy0> &poly_in,
                const tf::point_like<Dims, Policy1> &pt) -> bool {
  const auto &poly = tf::inject_plane(poly_in);
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
template <std::size_t Dims, typename Policy0, std::size_t V, typename Policy1>
auto intersects(const tf::point_like<Dims, Policy0> &pt,
                const tf::polygon<V, Policy1> &poly) -> bool {
  return intersects(poly, pt);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t V, typename Policy0, typename RealT, std::size_t Dims>
auto intersects(const tf::polygon<V, Policy0> &poly_in,
                const tf::ray<RealT, Dims> &ray) -> bool {
  const auto &poly = tf::inject_plane(poly_in);
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
template <typename RealT, std::size_t Dims, std::size_t V, typename Policy0>
auto intersects(const tf::ray<RealT, Dims> &ray,
                const tf::polygon<V, Policy0> &poly) {
  return intersects(poly, ray);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t V, typename Policy0, typename RealT, std::size_t Dims>
auto intersects(const tf::polygon<V, Policy0> &poly_in,
                const tf::line<RealT, Dims> &line) -> bool {
  const auto &poly = tf::inject_plane(poly_in);
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
template <typename RealT, std::size_t Dims, std::size_t V, typename Policy0>
auto intersects(const tf::line<RealT, Dims> &line,
                const tf::polygon<V, Policy0> &poly) {
  return intersects(poly, line);
}
/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t V, typename Policy0, typename Policy1>
auto intersects(const tf::polygon<V, Policy0> &poly_in,
                const tf::segment<Policy1> &seg1) -> bool {
  const auto &poly = tf::inject_plane(poly_in);
  auto ray = tf::make_ray_between_points(seg1[0], seg1[1]);
  using RealT = tf::common_value<decltype(poly[0][0]), decltype(seg1[0][0])>;
  return tf::ray_cast(ray, poly, tf::make_ray_config(RealT(0), RealT(1)));
}
/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <typename Policy, std::size_t V, typename Policy0>
auto intersects(const tf::segment<Policy> &seg,
                const tf::polygon<V, Policy0> &poly) {
  return intersects(poly, seg);
}

/// @ingroup geometry
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t V0, typename Policy0, std::size_t V1, typename Policy1>
auto intersects(const tf::polygon<V0, Policy0> &poly_in0,
                const tf::polygon<V1, Policy1> &poly_in1) -> bool {
  const auto &poly0 = tf::inject_plane(poly_in0);

  std::size_t size = poly0.size();
  std::size_t prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++) {
    if (intersects(poly0, tf::make_segment_between_points(poly_in1[prev],
                                                          poly_in1[i])))
      return true;
  }
  const auto &poly1 = tf::inject_plane(poly_in1);

  size = poly1.size();
  prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++) {
    if (intersects(poly1,
                   tf::make_segment_between_points(poly0[prev], poly0[i])))
      return true;
  }
  return false;
}

constexpr auto intersects_f =
    [](const auto &obj0,
       const auto &obj1) -> decltype(tf::intersects(obj0, obj1)) {
  return tf::intersects(obj0, obj1);
};

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::form<Dims, Policy0> &form0,
                const tf::form<Dims, Policy1> &form1) -> bool {
  return tf::search(form0, form1, tf::intersects_f, tf::intersects_f);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::point_like<Dims, Policy1> &obj) -> bool {
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj); },
      tf::intersects_f);
}

template <std::size_t Dims, typename Policy0, std::size_t V, typename Policy1>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::polygon<V, Policy1> &obj) -> bool {
  auto obj_aabb = tf::aabb_from(obj);
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj_aabb); },
      [&](const auto &other) { return tf::intersects(other, obj); });
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::segment<Policy1> &obj) -> bool {
  auto ray = tf::make_ray_between_points(obj[0], obj[1]);
  using real_t = tf::value_type<decltype(ray.origin)>;
  return tf::ray_cast(ray, form, tf::make_ray_config(real_t(0), real_t(1)));
}

template <std::size_t Dims, typename Policy0, typename RealT>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::ray<RealT, Dims> &obj) -> bool {
  return tf::ray_cast(obj, form);
}

template <std::size_t Dims, typename Policy0, typename RealT>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::line<RealT, Dims> &obj) -> bool {
  auto ray = tf::make_ray(obj.origin, obj.direction);
  using real_t = tf::value_type<decltype(ray.origin)>;
  return tf::ray_cast(ray, form,
                      tf::make_ray_config(-std::numeric_limits<real_t>::max(),
                                          std::numeric_limits<real_t>::max()));
}

} // namespace tf

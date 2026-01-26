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
#include "./base/local_box_intersects.hpp"
#include "./base/obb_intersects_primitive.hpp"
#include "./classify.hpp"
#include "./closest_point_parametric.hpp"
#include "./dot.hpp"
#include "./interval.hpp"
#include "./line.hpp"
#include "./obb_intersects_obb.hpp"
#include "./obb_like.hpp"
#include "./obbrss_like.hpp"
#include "./point_like.hpp"
#include "./polygon.hpp"
#include "./ray.hpp"
#include "./ray_cast.hpp"
#include "./segment.hpp"

namespace tf {

template <typename T0, typename T1>
auto intersects(const interval<T0> &r0, const interval<T1> &r1) -> bool {
  using RealT = std::common_type_t<T0, T1>;
  return !(r1.max + tf::epsilon<RealT> < r0.min ||
           r0.max + tf::epsilon<RealT> < r1.min);
}

/// @ingroup core_queries
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between
/// two AABBs.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const aabb_like<Dims, Policy0> &a,
                const aabb_like<Dims, Policy1> &b) -> bool {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  for (std::size_t i = 0; i < Dims; ++i) {
    if (a.max[i] + std::numeric_limits<RealT>::epsilon() < b.min[i] ||
        b.max[i] + std::numeric_limits<RealT>::epsilon() < a.min[i])
      return false;
  }
  return true;
}

/// @ingroup core_queries
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

/// @ingroup core_queries
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.

template <std::size_t N, typename T0, typename T1>
auto intersects(const point_like<N, T0> &point, const aabb_like<N, T1> &box)
    -> bool {
  using RealT = tf::coordinate_type<T0, T1>;
  for (std::size_t i = 0; i < N; ++i) {
    if (point[i] + std::numeric_limits<RealT>::epsilon() < box.min[i] ||
        point[i] - std::numeric_limits<RealT>::epsilon() > box.max[i])
      return false;
  }
  return true;
}

/// @ingroup core_queries
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

/// @ingroup core_queries
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

/// @ingroup core_queries
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
/// @ingroup core_queries
/// @brief Check whether two geometric primitives intersect within epsilon.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t N, typename T0, typename T1>
auto intersects(const point_like<N, T0> &v0, const point_like<N, T1> &v1,
                tf::coordinate_type<T0, T1> epsilon) -> bool {
  return (v0 - v1).length2() < epsilon;
}
/// @ingroup core_queries
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t N, typename T0, typename T1>
auto intersects(const tf::point_like<N, T0> &v0,
                const tf::point_like<N, T1> &v1) -> bool {
  return (v0 - v1).length2() < tf::epsilon2<tf::coordinate_type<T0, T1>>;
}

/// @ingroup core_queries
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
  return d2 < tf::epsilon2<decltype(d2)>;
}

/// @ingroup core_queries
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

/// @ingroup core_queries
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
  return d2 < tf::epsilon2<decltype(d2)>;
}

/// @ingroup core_queries
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

/// @ingroup core_queries
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
  return d2 < tf::epsilon2<decltype(d2)>;
}

template <typename T0, typename T1>
auto intersects(const tf::segment<2, T0> &s, const tf::point_like<2, T1> &v1) {
  return tf::classify(v1, s) == tf::sidedness::on_boundary;
}

/// @ingroup core_queries
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

template <typename T0, typename T1>
auto intersects(const tf::point_like<2, T0> &v0, const tf::segment<2, T1> &s) {
  return intersects(s, v0);
}

/// @ingroup core_queries
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
  return d2 < tf::epsilon2<decltype(d2)>;
}

/// @ingroup core_queries
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
  return d2 < tf::epsilon2<decltype(d2)>;
}

/// @ingroup core_queries
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
  return d2 < tf::epsilon2<decltype(d2)>;
}
/// @ingroup core_queries
/// @brief Check whether a ray and line intersect.
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::ray_like<Dims, Policy0> &r0,
                const tf::line_like<Dims, Policy1> &l1) {
  return intersects(l1, r0);
}

/// @ingroup core_queries
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
  auto [t0, t1] = tf::closest_point_parametric(r0, s1);
  auto pt0 = r0.origin + t0 * r0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  auto d2 = (pt0 - pt1).length2();
  return d2 < tf::epsilon2<decltype(d2)>;
}

/// @ingroup core_queries
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
  auto [t0, t1] = tf::closest_point_parametric(l0, s1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  auto d2 = (pt0 - pt1).length2();
  return d2 < tf::epsilon2<decltype(d2)>;
}

/// @ingroup core_queries
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
  auto [t0, t1] = tf::closest_point_parametric(s0, l1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  auto d2 = (pt0 - pt1).length2();
  return d2 < tf::epsilon2<decltype(d2)>;
}

/// @ingroup core_queries
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
  auto [t0, t1] = tf::closest_point_parametric(s0, r1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = r1.origin + t1 * r1.direction;
  auto d2 = (pt0 - pt1).length2();
  return d2 < tf::epsilon2<decltype(d2)>;
}

/// @ingroup core_queries
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
  auto [t0, t1] = tf::closest_point_parametric(s0, s1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  auto d2 = (pt0 - pt1).length2();
  return d2 < tf::epsilon2<decltype(d2)>;
}

template <typename T0, typename T1>
auto intersects(const tf::segment<2, T0> &s0, const tf::segment<2, T1> &s1)
    -> bool {
  using tf::classify;
  using tf::sidedness;

  const auto &a = s0[0];
  const auto &b = s0[1];
  const auto &c = s1[0];
  const auto &d = s1[1];

  const auto cd = tf::make_segment_between_points(c, d);
  const auto ab = tf::make_segment_between_points(a, b);

  const auto s_ac = classify(a, cd);
  const auto s_bc = classify(b, cd);
  const auto s_ca = classify(c, ab);
  const auto s_da = classify(d, ab);

  if (s_ac == sidedness::on_boundary || s_bc == sidedness::on_boundary ||
      s_ca == sidedness::on_boundary || s_da == sidedness::on_boundary) {
    return true;
  }

  const bool straddle1 = (s_ac != s_bc) && s_ac != sidedness::on_boundary &&
                         s_bc != sidedness::on_boundary;

  const bool straddle2 = (s_ca != s_da) && s_ca != sidedness::on_boundary &&
                         s_da != sidedness::on_boundary;

  return straddle1 && straddle2;
}

/// @ingroup core_queries
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
  return std::abs(d) < tf::epsilon<decltype(d)> &&
         tf::contains_coplanar_point(poly, c_pt);
}

template <typename Policy0, typename Policy1>
auto intersects(const tf::polygon<2, Policy0> &poly_in,
                const tf::point_like<2, Policy1> &pt) -> bool {
  return tf::contains_coplanar_point(poly_in, pt);
}

/// @ingroup core_queries
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

template <typename Policy0, typename Policy1>
auto intersects(const tf::point_like<2, Policy0> &pt,
                const tf::polygon<2, Policy1> &poly) -> bool {
  return intersects(poly, pt);
}

namespace core {
template <typename Policy0, std::size_t Dims, typename Policy>
auto intersects_coplanar(const tf::polygon<Dims, Policy0> &poly_in,
                         const tf::ray_like<Dims, Policy> &ray) -> bool {
  if (tf::intersects(poly_in, ray.origin))
    return true;
  std::size_t size = poly_in.size();
  auto prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++)
    if (tf::intersects(
            tf::make_segment_between_points(poly_in[prev], poly_in[i]), ray))
      return true;
  return false;
}
} // namespace core

template <typename Policy0, std::size_t Dims, typename Policy>
auto intersects(const tf::polygon<Dims, Policy0> &poly_in,
                const tf::ray_like<Dims, Policy> &ray) -> bool {
  const auto &poly = tf::tag_plane(poly_in);
  auto result = tf::ray_cast(ray, poly);
  if (result.status != tf::intersect_status::coplanar)
    return result;
  else
    return core::intersects_coplanar(poly_in, ray);
}

template <std::size_t Dims, typename Policy, typename Policy0>
auto intersects(const tf::ray_like<Dims, Policy> &ray,
                const tf::polygon<Dims, Policy0> &poly) {
  return intersects(poly, ray);
}

template <typename Policy0, typename Policy>
auto intersects(const tf::polygon<2, Policy0> &poly_in,
                const tf::ray_like<2, Policy> &ray) -> bool {
  return core::intersects_coplanar(poly_in, ray);
}

template <typename Policy0, typename Policy>
auto intersects(const tf::ray_like<2, Policy> &ray,
                const tf::polygon<2, Policy0> &poly_in) -> bool {
  return intersects(poly_in, ray);
}

namespace core {
template <typename Policy0, std::size_t Dims, typename Policy>
auto intersects_coplanar(const tf::polygon<Dims, Policy0> &poly_in,
                         const tf::line_like<Dims, Policy> &line) -> bool {
  std::size_t size = poly_in.size();
  auto prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++)
    if (tf::intersects(
            tf::make_segment_between_points(poly_in[prev], poly_in[i]), line))
      return true;
  return false;
}
} // namespace core
template <typename Policy0, std::size_t Dims, typename Policy>
auto intersects(const tf::polygon<Dims, Policy0> &poly_in,
                const tf::line_like<Dims, Policy> &line) -> bool {
  using RealT = tf::coordinate_type<Policy0, Policy>;
  const auto &poly = tf::tag_plane(poly_in);
  auto result =
      tf::ray_cast(tf::make_ray(line.origin, line.direction), poly,
                   tf::make_ray_config(-std::numeric_limits<RealT>::max(),
                                       std::numeric_limits<RealT>::max()));
  if (result.status != tf::intersect_status::coplanar)
    return result;
  else
    return core::intersects_coplanar(poly_in, line);
}

template <std::size_t Dims, typename Policy, typename Policy0>
auto intersects(const tf::line_like<Dims, Policy> &line,
                const tf::polygon<Dims, Policy0> &poly) {
  return intersects(poly, line);
}

template <typename Policy0, typename Policy>
auto intersects(const tf::polygon<2, Policy0> &poly_in,
                const tf::line_like<2, Policy> &line) -> bool {
  return core::intersects_coplanar(poly_in, line);
}

template <typename Policy0, typename Policy>
auto intersects(const tf::line_like<2, Policy> &line,
                const tf::polygon<2, Policy0> &poly_in) -> bool {
  return intersects(poly_in, line);
}

namespace core {
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects_coplanar(const tf::polygon<Dims, Policy0> &poly_in,
                         const tf::segment<Dims, Policy1> &seg1) -> bool {
  if (tf::contains_point(poly_in, seg1[0]) ||
      tf::contains_point(poly_in, seg1[1]))
    return true;
  std::size_t size = poly_in.size();
  auto prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++)
    if (tf::intersects(
            tf::make_segment_between_points(poly_in[prev], poly_in[i]), seg1))
      return true;
  return false;
}
} // namespace core

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::polygon<Dims, Policy0> &poly_in,
                const tf::segment<Dims, Policy1> &seg1) -> bool {
  const auto &poly = tf::tag_plane(poly_in);
  auto ray = tf::make_ray_between_points(seg1[0], seg1[1]);
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  auto result =
      tf::ray_cast(ray, poly, tf::make_ray_config(RealT(0), RealT(1)));
  if (result.status != tf::intersect_status::coplanar)
    return result;
  else
    return tf::core::intersects_coplanar(poly_in, seg1);
}

template <typename Policy0, typename Policy1>
auto intersects(const tf::polygon<2, Policy0> &poly_in,
                const tf::segment<2, Policy1> &seg1) -> bool {
  return tf::core::intersects_coplanar(poly_in, seg1);
}
/// @ingroup core_queries
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

template <typename Policy, typename Policy0>
auto intersects(const tf::segment<2, Policy> &seg,
                const tf::polygon<2, Policy0> &poly) {
  return intersects(poly, seg);
}

/// @ingroup core_queries
/// @brief Check whether two geometric primitives intersect.
///
/// This overload of `intersects` checks for intersection between specific
/// types.
///
/// @return `true` if the primitives intersect; otherwise `false`

namespace core {
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects_coplanar(const tf::polygon<Dims, Policy0> &poly0,
                         const tf::polygon<Dims, Policy1> &poly1) -> bool {

  std::size_t size0 = poly0.size();
  std::size_t prev0 = size0 - 1;
  std::size_t size1 = poly1.size();
  std::size_t prev1 = size1 - 1;
  for (std::size_t i = 0; i < size0; prev0 = i++) {
    auto seg0 = tf::make_segment_between_points(poly0[prev0], poly0[i]);
    for (std::size_t j = 0; j < size1; prev1 = j++) {
      auto seg1 = tf::make_segment_between_points(poly1[prev1], poly1[j]);
      if (intersects(seg0, seg1))
        return true;
    }
  }
  return intersects(poly0, poly1[0]) || intersects(poly1, poly0[0]);
}
} // namespace core

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::polygon<Dims, Policy0> &poly_in0,
                const tf::polygon<Dims, Policy1> &poly_in1) -> bool {

  const auto &poly0 = tf::tag_plane(poly_in0);
  const auto &poly1 = tf::tag_plane(poly_in1);
  auto dot = std::abs(tf::dot(poly0.normal(), poly1.normal()));
  if (1 - dot < tf::epsilon<decltype(dot)>) {
    if (intersects(poly0.plane(), poly_in1[0]))
      return core::intersects_coplanar(poly_in0, poly_in1);
    else
      return false;
  }

  std::size_t size = poly0.size();
  std::size_t prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++) {
    if (intersects(poly0, tf::make_segment_between_points(poly_in1[prev],
                                                          poly_in1[i])))
      return true;
  }

  size = poly1.size();
  prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++) {
    if (intersects(poly1,
                   tf::make_segment_between_points(poly0[prev], poly0[i])))
      return true;
  }
  return false;
}

template <typename Policy0, typename Policy1>
auto intersects(const tf::polygon<2, Policy0> &poly0,
                const tf::polygon<2, Policy1> &poly1) -> bool {
  return core::intersects_coplanar(poly0, poly1);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::plane_like<Dims, Policy0> &plane,
                const tf::point_like<Dims, Policy1> &pt) -> bool {
  auto d = tf::dot(plane.normal, pt) + plane.d;
  ;
  return std::abs(d) < tf::epsilon<decltype(d)>;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::point_like<Dims, Policy0> &pt,
                const tf::plane_like<Dims, Policy1> &plane) -> bool {
  return intersects(plane, pt);
}

/// @ingroup core_queries
/// @brief Check whether two planes intersect.
///
/// Two planes intersect if they are not parallel, or if they are coplanar.
///
/// @return `true` if the planes intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::plane_like<Dims, Policy0> &plane0,
                const tf::plane_like<Dims, Policy1> &plane1) -> bool {
  using T = tf::coordinate_type<Policy0, Policy1>;

  auto dot_n = tf::dot(plane0.normal, plane1.normal);

  // If not parallel (|dot| < 1), they intersect along a line
  if (std::abs(dot_n) < T(1) - tf::epsilon<T>)
    return true;

  // Parallel - check if coplanar
  // Point on plane0: p = -d0 * n0
  // Check if on plane1: dot(n1, p) + d1 = -d0 * dot(n0,n1) + d1
  T check = plane1.d - plane0.d * dot_n;
  return std::abs(check) < tf::epsilon<T>;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::plane_like<Dims, Policy0> &plane,
                const tf::ray_like<Dims, Policy1> &ray) -> bool {
  return core::does_intersect_any(tf::ray_cast(ray, plane));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::ray_like<Dims, Policy0> &ray,
                const tf::plane_like<Dims, Policy1> &plane) -> bool {
  return core::does_intersect_any(tf::ray_cast(ray, plane));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::plane_like<Dims, Policy0> &plane,
                const tf::line_like<Dims, Policy1> &line) -> bool {
  return core::does_intersect_any(tf::ray_cast(
      tf::make_ray_like(line.origin, line.direction), plane,
      tf::make_ray_config(
          std::numeric_limits<tf::coordinate_type<Policy0, Policy1>>::lowest(),
          std::numeric_limits<tf::coordinate_type<Policy0, Policy1>>::max())));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::line_like<Dims, Policy0> &line,
                const tf::plane_like<Dims, Policy1> &plane) -> bool {
  return intersects(plane, line);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::plane_like<Dims, Policy0> &plane,
                const tf::segment<Dims, Policy1> &seg) -> bool {
  return core::does_intersect_any(tf::ray_cast(
      tf::make_ray_like(seg[0], seg[1] - seg[0]), plane,
      tf::make_ray_config(tf::coordinate_type<Policy0, Policy1>(0),
                          tf::coordinate_type<Policy0, Policy1>(1))));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::segment<Dims, Policy0> &seg,
                const tf::plane_like<Dims, Policy1> &plane) -> bool {
  return intersects(plane, seg);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::polygon<Dims, Policy0> &poly,
                const tf::plane_like<Dims, Policy1> &plane) {
  bool positive = false;
  bool negative = false;
  for (const auto &pt : poly) {
    auto d = tf::dot(plane.normal, pt) + plane.d;
    if (std::abs(d) < tf::epsilon<decltype(d)>)
      return true;
    bool test = d > 0;
    positive |= test;
    negative |= !test;
    if (positive && negative)
      return true;
  }
  return false;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::plane_like<Dims, Policy0> &plane,
                const tf::polygon<Dims, Policy1> &poly) -> bool {
  return intersects(poly, plane);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::aabb_like<Dims, Policy0> &bbox,
                const tf::plane_like<Dims, Policy1> &plane) {
  tf::coordinate_type<Policy0, Policy1> n_min = 0;
  decltype(n_min) n_max = 0;
  for (std::size_t i = 0; i < Dims; ++i) {
    std::array<decltype(n_min), 2> ds{bbox.min[i], bbox.max[i]};
    bool test = plane.normal[i] < 0;
    n_min += ds[test] * plane.normal[i];
    n_max += ds[!test] * plane.normal[i];
  }
  n_min += plane.d;
  n_max += plane.d;
  return n_min <= tf::epsilon<decltype(n_min)> &&
         n_max >= -tf::epsilon<decltype(n_max)>;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::plane_like<Dims, Policy0> &plane,
                const tf::aabb_like<Dims, Policy1> &bbox) {
  return intersects(bbox, plane);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::aabb_like<Dims, Policy0> &bbox,
                const tf::polygon<Dims, Policy1> &poly) {
  std::size_t size = poly.size();
  std::size_t prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++) {
    if (intersects(bbox, tf::make_segment_between_points(poly[prev], poly[i])))
      return true;
  }
  return false;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::polygon<Dims, Policy0> &poly,
                const tf::aabb_like<Dims, Policy1> &bbox) {
  return intersects(bbox, poly);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::ray_like<Dims, Policy0> &ray,
                const tf::aabb_like<Dims, Policy1> &bbox) -> bool {
  return tf::ray_cast(ray, bbox);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::aabb_like<Dims, Policy0> &bbox,
                const tf::ray_like<Dims, Policy1> &ray) {
  return intersects(ray, bbox);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::line_like<Dims, Policy0> &line,
                const tf::aabb_like<Dims, Policy1> &bbox) -> bool {
  auto ray = tf::make_ray_like(line.origin, line.direction);
  using real_t = tf::coordinate_type<Policy0, Policy1>;
  return tf::ray_cast(ray, bbox,
                      tf::make_ray_config(std::numeric_limits<real_t>::lowest(),
                                          std::numeric_limits<real_t>::max()));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::aabb_like<Dims, Policy0> &bbox,
                const tf::line_like<Dims, Policy1> &line) {
  return intersects(line, bbox);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::segment<Dims, Policy0> &seg,
                const tf::aabb_like<Dims, Policy1> &bbox) -> bool {
  auto ray = tf::make_ray_between_points(seg[0], seg[1]);
  using real_t = tf::coordinate_type<Policy0, Policy1>;
  return tf::ray_cast(ray, bbox, tf::make_ray_config(real_t(0), real_t(1)));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::aabb_like<Dims, Policy0> &bbox,
                const tf::segment<Dims, Policy1> &seg) {
  return intersects(seg, bbox);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obb_like<Dims, Policy0> &obb0,
                const tf::obb_like<Dims, Policy1> &obb1) -> bool {
  return core::obb_intersects_obb(obb0, obb1);
}

/// @ingroup core_queries
/// @brief Check whether an OBB and AABB intersect.
///
/// Uses the Separating Axis Theorem (SAT) testing 2*Dims axes.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obb_like<Dims, Policy0> &obb,
                const tf::aabb_like<Dims, Policy1> &bbox) -> bool {
  using T = tf::coordinate_type<Policy0, Policy1>;

  for (std::size_t a = 0; a < Dims; ++a) {
    // Test OBB axis[a]: project AABB onto it
    {
      T proj_min = 0, proj_max = 0;
      for (std::size_t i = 0; i < Dims; ++i) {
        T d_min = (bbox.min[i] - obb.origin[i]) * obb.axes[a][i];
        T d_max = (bbox.max[i] - obb.origin[i]) * obb.axes[a][i];
        proj_min += std::min(d_min, d_max);
        proj_max += std::max(d_min, d_max);
      }
      if (proj_max < 0 || proj_min > obb.extent[a])
        return false;
    }

    // Test AABB axis[a]: project OBB onto it
    {
      T proj_min = obb.origin[a], proj_max = obb.origin[a];
      for (std::size_t i = 0; i < Dims; ++i) {
        T d = obb.axes[i][a] * obb.extent[i];
        proj_min += std::min(d, T{0});
        proj_max += std::max(d, T{0});
      }
      if (proj_max + std::numeric_limits<T>::epsilon() < bbox.min[a] ||
          proj_min - std::numeric_limits<T>::epsilon() > bbox.max[a])
        return false;
    }
  }
  return true;
}

/// @ingroup core_queries
/// @brief Check whether an AABB and OBB intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::aabb_like<Dims, Policy0> &bbox,
                const tf::obb_like<Dims, Policy1> &obb) -> bool {
  return intersects(obb, bbox);
}

/// @ingroup core_queries
/// @brief Check whether an OBB contains a point.
///
/// @return `true` if the point is inside the OBB; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obb_like<Dims, Policy0> &obb,
                const tf::point_like<Dims, Policy1> &pt) -> bool {
  using T = tf::coordinate_type<Policy0, Policy1>;

  // Transform point to local coordinates
  auto diff = pt - obb.origin;
  std::array<T, Dims> local_pt;
  for (std::size_t i = 0; i < Dims; ++i) {
    local_pt[i] = tf::dot(diff, obb.axes[i]);
  }

  // Check if inside box in local coords
  std::array<T, Dims> extent;
  for (std::size_t i = 0; i < Dims; ++i) {
    extent[i] = obb.extent[i];
  }
  return tf::core::local_point_box_intersects(local_pt, extent);
}

/// @ingroup core_queries
/// @brief Check whether a point is inside an OBB.
///
/// @return `true` if the point is inside the OBB; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::point_like<Dims, Policy0> &pt,
                const tf::obb_like<Dims, Policy1> &obb) -> bool {
  return intersects(obb, pt);
}

/// @ingroup core_queries
/// @brief Check whether an OBB and ray intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obb_like<Dims, Policy0> &obb,
                const tf::ray_like<Dims, Policy1> &ray) -> bool {
  return core::obb_intersects_ray(obb, ray);
}

/// @ingroup core_queries
/// @brief Check whether a ray and OBB intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::ray_like<Dims, Policy0> &ray,
                const tf::obb_like<Dims, Policy1> &obb) -> bool {
  return intersects(obb, ray);
}

/// @ingroup core_queries
/// @brief Check whether an OBB and line intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obb_like<Dims, Policy0> &obb,
                const tf::line_like<Dims, Policy1> &line) -> bool {
  return core::obb_intersects_line(obb, line);
}

/// @ingroup core_queries
/// @brief Check whether a line and OBB intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::line_like<Dims, Policy0> &line,
                const tf::obb_like<Dims, Policy1> &obb) -> bool {
  return intersects(obb, line);
}

/// @ingroup core_queries
/// @brief Check whether an OBB and segment intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obb_like<Dims, Policy0> &obb,
                const tf::segment<Dims, Policy1> &seg) -> bool {
  return core::obb_intersects_segment(obb, seg);
}

/// @ingroup core_queries
/// @brief Check whether a segment and OBB intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::segment<Dims, Policy0> &seg,
                const tf::obb_like<Dims, Policy1> &obb) -> bool {
  return intersects(obb, seg);
}

/// @ingroup core_queries
/// @brief Check whether an OBB and plane intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obb_like<Dims, Policy0> &obb,
                const tf::plane_like<Dims, Policy1> &plane) -> bool {
  return core::obb_intersects_plane(obb, plane);
}

/// @ingroup core_queries
/// @brief Check whether a plane and OBB intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::plane_like<Dims, Policy0> &plane,
                const tf::obb_like<Dims, Policy1> &obb) -> bool {
  return intersects(obb, plane);
}

/// @ingroup core_queries
/// @brief Check whether an OBB and polygon intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obb_like<Dims, Policy0> &obb,
                const tf::polygon<Dims, Policy1> &poly) -> bool {
  return core::obb_intersects_polygon(obb, poly);
}

/// @ingroup core_queries
/// @brief Check whether a polygon and OBB intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::polygon<Dims, Policy0> &poly,
                const tf::obb_like<Dims, Policy1> &obb) -> bool {
  return intersects(obb, poly);
}

/// @ingroup core_queries
/// @brief Check whether two OBBRSSs intersect.
///
/// Uses the OBB parts for intersection testing (cheaper than RSS).
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obbrss_like<Dims, Policy0> &a,
                const tf::obbrss_like<Dims, Policy1> &b) -> bool {
  return intersects(tf::make_obb_like(a.obb_origin, a.axes, a.extent),
                    tf::make_obb_like(b.obb_origin, b.axes, b.extent));
}

/// @ingroup core_queries
/// @brief Check whether an OBBRSS and AABB intersect.
///
/// Uses the OBB part of OBBRSS for intersection testing.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obbrss_like<Dims, Policy0> &obbrss,
                const tf::aabb_like<Dims, Policy1> &bbox) -> bool {
  return intersects(
      tf::make_obb_like(obbrss.obb_origin, obbrss.axes, obbrss.extent), bbox);
}

/// @ingroup core_queries
/// @brief Check whether an AABB and OBBRSS intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::aabb_like<Dims, Policy0> &bbox,
                const tf::obbrss_like<Dims, Policy1> &obbrss) -> bool {
  return intersects(obbrss, bbox);
}

/// @ingroup core_queries
/// @brief Check whether an OBBRSS and OBB intersect.
///
/// Uses the OBB part of OBBRSS for intersection testing.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obbrss_like<Dims, Policy0> &obbrss,
                const tf::obb_like<Dims, Policy1> &obb) -> bool {
  return intersects(
      tf::make_obb_like(obbrss.obb_origin, obbrss.axes, obbrss.extent), obb);
}

/// @ingroup core_queries
/// @brief Check whether an OBB and OBBRSS intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obb_like<Dims, Policy0> &obb,
                const tf::obbrss_like<Dims, Policy1> &obbrss) -> bool {
  return intersects(obbrss, obb);
}

/// @ingroup core_queries
/// @brief Check whether an OBBRSS contains a point.
///
/// Uses the OBB part of OBBRSS for intersection testing.
///
/// @return `true` if the point is inside the OBBRSS; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obbrss_like<Dims, Policy0> &obbrss,
                const tf::point_like<Dims, Policy1> &pt) -> bool {
  return intersects(
      tf::make_obb_like(obbrss.obb_origin, obbrss.axes, obbrss.extent), pt);
}

/// @ingroup core_queries
/// @brief Check whether a point is inside an OBBRSS.
///
/// @return `true` if the point is inside the OBBRSS; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::point_like<Dims, Policy0> &pt,
                const tf::obbrss_like<Dims, Policy1> &obbrss) -> bool {
  return intersects(obbrss, pt);
}

/// @ingroup core_queries
/// @brief Check whether an OBBRSS and ray intersect.
///
/// Uses the OBB part of OBBRSS for intersection testing.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obbrss_like<Dims, Policy0> &obbrss,
                const tf::ray_like<Dims, Policy1> &ray) -> bool {
  return intersects(
      tf::make_obb_like(obbrss.obb_origin, obbrss.axes, obbrss.extent), ray);
}

/// @ingroup core_queries
/// @brief Check whether a ray and OBBRSS intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::ray_like<Dims, Policy0> &ray,
                const tf::obbrss_like<Dims, Policy1> &obbrss) -> bool {
  return intersects(obbrss, ray);
}

/// @ingroup core_queries
/// @brief Check whether an OBBRSS and line intersect.
///
/// Uses the OBB part of OBBRSS for intersection testing.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obbrss_like<Dims, Policy0> &obbrss,
                const tf::line_like<Dims, Policy1> &line) -> bool {
  return intersects(
      tf::make_obb_like(obbrss.obb_origin, obbrss.axes, obbrss.extent), line);
}

/// @ingroup core_queries
/// @brief Check whether a line and OBBRSS intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::line_like<Dims, Policy0> &line,
                const tf::obbrss_like<Dims, Policy1> &obbrss) -> bool {
  return intersects(obbrss, line);
}

/// @ingroup core_queries
/// @brief Check whether an OBBRSS and segment intersect.
///
/// Uses the OBB part of OBBRSS for intersection testing.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obbrss_like<Dims, Policy0> &obbrss,
                const tf::segment<Dims, Policy1> &seg) -> bool {
  return intersects(
      tf::make_obb_like(obbrss.obb_origin, obbrss.axes, obbrss.extent), seg);
}

/// @ingroup core_queries
/// @brief Check whether a segment and OBBRSS intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::segment<Dims, Policy0> &seg,
                const tf::obbrss_like<Dims, Policy1> &obbrss) -> bool {
  return intersects(obbrss, seg);
}

/// @ingroup core_queries
/// @brief Check whether an OBBRSS and plane intersect.
///
/// Uses the OBB part of OBBRSS for intersection testing.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obbrss_like<Dims, Policy0> &obbrss,
                const tf::plane_like<Dims, Policy1> &plane) -> bool {
  return intersects(
      tf::make_obb_like(obbrss.obb_origin, obbrss.axes, obbrss.extent), plane);
}

/// @ingroup core_queries
/// @brief Check whether a plane and OBBRSS intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::plane_like<Dims, Policy0> &plane,
                const tf::obbrss_like<Dims, Policy1> &obbrss) -> bool {
  return intersects(obbrss, plane);
}

/// @ingroup core_queries
/// @brief Check whether an OBBRSS and polygon intersect.
///
/// Uses the OBB part of OBBRSS for intersection testing.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::obbrss_like<Dims, Policy0> &obbrss,
                const tf::polygon<Dims, Policy1> &poly) -> bool {
  return intersects(
      tf::make_obb_like(obbrss.obb_origin, obbrss.axes, obbrss.extent), poly);
}

/// @ingroup core_queries
/// @brief Check whether a polygon and OBBRSS intersect.
///
/// @return `true` if the primitives intersect; otherwise `false`.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::polygon<Dims, Policy0> &poly,
                const tf::obbrss_like<Dims, Policy1> &obbrss) -> bool {
  return intersects(obbrss, poly);
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

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
#include "./closest_metric_point_pair.hpp"
#include "./closest_point_parametric.hpp"
#include "./metric_point.hpp"

namespace tf {

/// @ingroup core_queries
/// @brief Computes the closest point on the first object with its squared distance.
///
/// Returns a @ref tf::metric_point containing the closest point on the first
/// argument and the squared distance to the second argument.
///
/// @tparam Dims The dimensionality.
/// @tparam T0 The first object's policy.
/// @tparam T1 The second object's policy.
/// @param v0 The first object.
/// @param v1 The second object.
/// @return A @ref tf::metric_point with the closest point and squared distance.
template <std::size_t Dims, typename T0, typename T1>
auto closest_metric_point(const tf::vector_like<Dims, T0> &v0,
                          const tf::vector_like<Dims, T1> &v1) {
  return tf::make_metric_point((v0 - v1).length2(), v0);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename T0, typename T1>
auto closest_metric_point(const tf::point_like<Dims, T0> &v0,
                          const tf::point_like<Dims, T1> &v1) {
  return tf::make_metric_point((v0 - v1).length2(), v0);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy, typename T1>
auto closest_metric_point(const tf::line_like<Dims, Policy> &l,
                          const tf::point_like<Dims, T1> &v1) {
  auto t = tf::closest_point_parametric(l, v1);
  auto pt = l.origin + t * l.direction;
  return tf::make_metric_point((pt - v1).length2(), pt);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename T1, typename Policy>
auto closest_metric_point(const tf::point_like<Dims, T1> &v0,
                          const tf::line_like<Dims, Policy> &l) {
  auto t = tf::closest_point_parametric(l, v0);
  auto pt = l.origin + t * l.direction;
  return tf::make_metric_point((pt - v0).length2(), v0);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy, typename T1>
auto closest_metric_point(const tf::ray_like<Dims, Policy> &r,
                          const tf::point_like<Dims, T1> &v1) {
  auto t = tf::closest_point_parametric(r, v1);
  auto pt = r.origin + t * r.direction;
  return tf::make_metric_point((pt - v1).length2(), pt);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename T1, typename Policy>
auto closest_metric_point(const tf::point_like<Dims, T1> &v0,
                          const tf::ray_like<Dims, Policy> &r) {
  auto t = tf::closest_point_parametric(r, v0);
  auto pt = r.origin + t * r.direction;
  return tf::make_metric_point((pt - v0).length2(), v0);
}

/// @ingroup core_queries
/// @overload
template <typename T0, std::size_t Dims, typename T1>
auto closest_metric_point(const tf::segment<Dims, T0> &s,
                          const tf::point_like<Dims, T1> &v1) {
  auto t = tf::closest_point_parametric(s, v1);
  auto l = tf::make_line_between_points(s[0], s[1]);
  auto pt = l.origin + t * l.direction;
  return tf::make_metric_point((pt - v1).length2(), pt);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename T0, typename T1>
auto closest_metric_point(const tf::point_like<Dims, T0> &v0,
                          const tf::segment<Dims, T1> &s) {
  auto t = tf::closest_point_parametric(s, v0);
  auto l = tf::make_line_between_points(s[0], s[1]);
  auto pt = l.origin + t * l.direction;
  return tf::make_metric_point((pt - v0).length2(), v0);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point(const tf::line_like<Dims, Policy0> &l0,
                          const tf::line_like<Dims, Policy1> &l1) {
  auto [t0, t1] = tf::closest_point_parametric(l0, l1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  return tf::make_metric_point((pt0 - pt1).length2(), pt0);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point(const tf::ray_like<Dims, Policy0> &r0,
                          const tf::ray_like<Dims, Policy1> &r1) {
  auto [t0, t1] = tf::closest_point_parametric(r0, r1);
  auto pt0 = r0.origin + t0 * r0.direction;
  auto pt1 = r1.origin + t1 * r1.direction;
  return tf::make_metric_point((pt0 - pt1).length2(), pt0);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point(const tf::line_like<Dims, Policy0> &l0,
                          const tf::ray_like<Dims, Policy1> &r1) {
  auto [t0, t1] = tf::closest_point_parametric(l0, r1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = r1.origin + t1 * r1.direction;
  return tf::make_metric_point((pt0 - pt1).length2(), pt0);
}
/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point(const tf::ray_like<Dims, Policy0> &r0,
                          const tf::line_like<Dims, Policy1> &l1) {
  auto [t0, t1] = tf::closest_point_parametric(r0, l1);
  auto pt0 = r0.origin + t0 * r0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  return tf::make_metric_point((pt0 - pt1).length2(), pt0);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy, typename T>
auto closest_metric_point(const tf::ray_like<Dims, Policy> &r0,
                          const tf::segment<Dims, T> &s1) {
  auto l1 = tf::make_line_between_points(s1[0], s1[1]);
  auto [t0, t1] = tf::closest_point_parametric(r0, s1);
  auto pt0 = r0.origin + t0 * r0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  return tf::make_metric_point((pt0 - pt1).length2(), pt0);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy, typename T>
auto closest_metric_point(const tf::line_like<Dims, Policy> &l0,
                          const tf::segment<Dims, T> &s1) {
  auto l1 = tf::make_line_between_points(s1[0], s1[1]);
  auto [t0, t1] = tf::closest_point_parametric(l0, s1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  return tf::make_metric_point((pt0 - pt1).length2(), pt0);
}

/// @ingroup core_queries
/// @overload
template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point(const tf::segment<Dims, T> &s0,
                          const tf::line_like<Dims, Policy> &l1) {
  auto l0 = tf::make_line_between_points(s0[0], s0[1]);
  auto [t0, t1] = tf::closest_point_parametric(s0, l1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  return tf::make_metric_point((pt0 - pt1).length2(), pt0);
}

/// @ingroup core_queries
/// @overload
template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point(const tf::segment<Dims, T> &s0,
                          const tf::ray_like<Dims, Policy> &r1) {
  auto l0 = tf::make_line_between_points(s0[0], s0[1]);
  auto [t0, t1] = tf::closest_point_parametric(s0, r1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = r1.origin + t1 * r1.direction;
  return tf::make_metric_point((pt0 - pt1).length2(), pt0);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename T0, typename T1>
auto closest_metric_point(const tf::segment<Dims, T0> &s0,
                          const tf::segment<Dims, T1> &s1) {
  auto l0 = tf::make_line_between_points(s0[0], s0[1]);
  auto l1 = tf::make_line_between_points(s1[0], s1[1]);
  auto [t0, t1] = tf::closest_point_parametric(s0, s1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  return tf::make_metric_point((pt0 - pt1).length2(), pt0);
}

/// @ingroup core_queries
/// @overload
template <typename Policy0, std::size_t Dims, typename Policy1>
auto closest_metric_point(const tf::polygon<Dims, Policy0> &poly_in,
                          const tf::point_like<Dims, Policy1> &pt) {
  auto res = tf::closest_metric_point_pair(poly_in, pt);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy1, typename Policy0>
auto closest_metric_point(const tf::point_like<Dims, Policy1> &pt,
                          const tf::polygon<Dims, Policy0> &poly) {
  auto res = tf::closest_metric_point_pair(pt, poly);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <typename Policy0, std::size_t Dims, typename Policy>
auto closest_metric_point(const tf::polygon<Dims, Policy0> &poly_in,
                          const tf::line_like<Dims, Policy> &line) {
  auto res = tf::closest_metric_point_pair(poly_in, line);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy, typename Policy0>
auto closest_metric_point(const tf::line_like<Dims, Policy> &line,
                          const tf::polygon<Dims, Policy0> &poly) {
  auto res = tf::closest_metric_point_pair(line, poly);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <typename Policy0, std::size_t Dims, typename Policy>
auto closest_metric_point(const tf::polygon<Dims, Policy0> &poly_in,
                          const tf::ray_like<Dims, Policy> &ray) {
  auto res = tf::closest_metric_point_pair(poly_in, ray);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy, typename Policy0>
auto closest_metric_point(const tf::ray_like<Dims, Policy> &ray,
                          const tf::polygon<Dims, Policy0> &poly) {
  auto res = tf::closest_metric_point_pair(ray, poly);
  return tf::make_metric_point(res.metric, res.first);
}
/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point(const tf::polygon<Dims, Policy0> &poly_in,
                          const tf::segment<Dims, Policy1> &seg1) {
  auto res = tf::closest_metric_point_pair(poly_in, seg1);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <typename Policy, std::size_t Dims, typename Policy0>
auto closest_metric_point(const tf::segment<Dims, Policy> &seg,
                          const tf::polygon<Dims, Policy0> &poly) {
  auto res = tf::closest_metric_point_pair(seg, poly);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point(const tf::polygon<Dims, Policy0> &poly_in0,
                          const tf::polygon<Dims, Policy1> &poly_in1) {
  auto res = tf::closest_metric_point_pair(poly_in0, poly_in1);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point(const tf::plane_like<Dims, T> &p,
                          const tf::point_like<Dims, Policy> &pt) {
  auto res = closest_metric_point_pair(p, pt);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point(const tf::segment<Dims, T> &s0,
                          const tf::plane_like<Dims, Policy> &p1) {
  auto res = closest_metric_point_pair(s0, p1);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point(const tf::plane_like<Dims, T> &o0,
                          const tf::segment<Dims, Policy> &o1) {
  auto res = closest_metric_point_pair(o0, o1);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point(const tf::ray_like<Dims, T> &o0,
                          const tf::plane_like<Dims, Policy> &p1) {
  auto res = closest_metric_point_pair(o0, p1);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point(const tf::plane_like<Dims, T> &o0,
                          const tf::ray_like<Dims, Policy> &o1) {
  auto res = closest_metric_point_pair(o0, o1);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point(const tf::line_like<Dims, T> &o0,
                          const tf::plane_like<Dims, Policy> &p1) {
  auto res = closest_metric_point_pair(o0, p1);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point(const tf::plane_like<Dims, T> &o0,
                          const tf::line_like<Dims, Policy> &o1) {
  auto res = closest_metric_point_pair(o0, o1);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point(const tf::point_like<Dims, T> &pt,
                          const tf::plane_like<Dims, Policy> &p) {
  auto res = closest_metric_point_pair(pt, p);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point(const tf::plane_like<Dims, Policy0> &p0,
                          const tf::plane_like<Dims, Policy1> &p1) {
  auto res = closest_metric_point_pair(p0, p1);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point(const tf::polygon<Dims, Policy0> &poly,
                          const tf::plane_like<Dims, Policy1> &plane) {
  auto res = closest_metric_point_pair(poly, plane);
  return tf::make_metric_point(res.metric, res.first);
}

/// @ingroup core_queries
/// @overload
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point(const tf::plane_like<Dims, Policy0> &plane,
                          const tf::polygon<Dims, Policy1> &poly) {
  auto res = closest_metric_point_pair(plane, poly);
  return tf::make_metric_point(res.metric, res.first);
}

} // namespace tf

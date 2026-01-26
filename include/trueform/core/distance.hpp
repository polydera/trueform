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
#include "./base/local_rectangle_distance.hpp"
#include "./closest_metric_point.hpp"
#include "./coordinate_type.hpp"
#include "./dot.hpp"
#include "./line_like.hpp"
#include "./obb_like.hpp"
#include "./obbrss_like.hpp"
#include "./plane_like.hpp"
#include "./point_like.hpp"
#include "./ray_like.hpp"
#include "./rss_like.hpp"
#include "./segment.hpp"
#include "./sphere_like.hpp"
#include "./sqrt.hpp"

namespace tf {

/// @ingroup core_queries
/// @brief Computes the squared Euclidean distance between two vectors.
/// @tparam N The dimensionality.
/// @tparam T0 The vector policy
/// @tparam T1 The vector policy
/// @param a First vector.
/// @param b Second vector.
/// @return Squared distance between a and b.
template <std::size_t N, typename T0, typename T1>
auto distance2(const vector_like<N, T0> &a, const vector_like<N, T1> &b)
    -> tf::coordinate_type<T0, T1> {
  return (a - b).length2();
}

/// @ingroup core_queries
/// @brief Computes the Euclidean distance between two vectors.
/// @tparam N The dimensionality.
/// @tparam T0 The vector policy
/// @tparam T1 The vector policy
/// @param a First vector.
/// @param b Second vector.
/// @return Distance between a and b.
template <std::size_t N, typename T0, typename T1>
auto distance(const vector_like<N, T0> &a, const vector_like<N, T1> &b)
    -> tf::coordinate_type<T0, T1> {
  return tf::sqrt(distance2(a, b));
}

/// @ingroup core_queries
/// @brief Computes the squared Euclidean distance between two points.
/// @tparam N The dimensionality.
/// @tparam T0 The point policy
/// @tparam T1 The point policy
/// @param a First point.
/// @param b Second point.
/// @return Squared distance between a and b.
template <std::size_t N, typename T0, typename T1>
auto distance2(const point_like<N, T0> &a, const point_like<N, T1> &b)
    -> tf::coordinate_type<T0, T1> {
  return (a - b).length2();
}

/// @ingroup core_queries
/// @brief Computes the squared Euclidean distance between two points.
/// @tparam N The dimensionality.
/// @tparam T0 The point policy
/// @tparam T1 The point policy
/// @param a First point.
/// @param b Second point.
/// @return Squared distance between a and b.
template <std::size_t N, typename T0, typename T1>
auto distance(const point_like<N, T0> &a, const point_like<N, T1> &b)
    -> tf::coordinate_type<T0, T1> {
  return tf::sqrt(distance2(a, b));
}

/// @ingroup core_queries
/// @brief Computes the squared distance between two AABBs.
/// The result is 0 if they overlap.
/// @tparam T The scalar type.
/// @tparam N The dimensionality.
/// @param a First AABB.
/// @param b Second AABB.
/// @return Squared distance between AABBs.
template <std::size_t N, typename T0, typename T1>
auto distance2(const aabb_like<N, T0> &a, const aabb_like<N, T1> &b) {
  using T = tf::coordinate_type<T0, T1>;
  T dist2 = T{};
  for (std::size_t i = 0; i < N; ++i) {
    const auto d1 =
        std::max(a.min[i] - b.max[i], decltype(a.min[i] - b.max[i]){0});
    auto d2 = std::max(b.min[i] - a.max[i], decltype(a.min[i] - b.max[i]){0});
    d2 *= d1 == 0;
    dist2 += d1 * d1 + d2 * d2;
  }
  return dist2;
}

/// @ingroup core_queries
/// @brief Computes the distance between two AABBs.
/// @tparam T The scalar type.
/// @tparam N The dimensionality.
/// @param a First AABB.
/// @param b Second AABB.
/// @return Distance between AABBs.
template <std::size_t N, typename T0, typename T1>
auto distance(const aabb_like<N, T0> &a, const aabb_like<N, T1> &b) {
  return tf::sqrt(distance2(a, b));
}

/// @ingroup core_queries
/// @brief Computes the squared distance from a point to an AABB.
/// @tparam N The dimensionality.
/// @tparam T The aabb value type
/// @tparam T1 The point policy
/// @param _bbox The AABB.
/// @param _point The point.
/// @return Squared distance from point to AABB.
template <std::size_t N, typename T0, typename T1>
auto distance2(const aabb_like<N, T0> &_bbox, const point_like<N, T1> &_point) {
  tf::coordinate_type<T0, T1> dist2{};
  const auto &min = _bbox.min;
  const auto &max = _bbox.max;
  for (std::size_t i = 0; i < N; ++i) {
    auto outside_low =
        std::max(min[i] - _point[i], decltype(min[i] - _point[i]){0});
    auto outside_high =
        std::max(_point[i] - max[i], decltype(_point[i] - max[i]){0});
    outside_high *= outside_low == 0;
    dist2 += outside_low * outside_low + outside_high * outside_high;
  }
  return dist2;
}

/// @ingroup core_queries
/// @brief Computes the squared distance from a point to an AABB (reverse
/// argument order).
template <std::size_t N, typename T0, typename T1>
auto distance2(const point_like<N, T0> &_point, const aabb_like<N, T1> &_bbox) {
  return distance2(_bbox, _point);
}

/// @ingroup core_queries
/// @brief Computes the distance from a point to an AABB.
template <std::size_t N, typename T0, typename T1>
auto distance(const aabb_like<N, T0> &_bbox, const point_like<N, T1> &_point) {
  return tf::sqrt(distance2(_bbox, _point));
}

/// @ingroup core_queries
/// @brief Computes the distance from a point to an AABB (reverse argument
/// order).
template <std::size_t N, typename T0, typename T1>
auto distance(const point_like<N, T0> &_point, const aabb_like<N, T1> &_bbox) {
  return tf::sqrt(distance2(_bbox, _point));
}

/// @ingroup core_queries
/// @brief Computes the signed distance from a point to a plane.
///
/// Returns positive distance if the point is on the side of the plane
/// indicated by the normal, negative otherwise.
///
/// @tparam N The dimensionality.
/// @tparam T0 The plane policy type.
/// @tparam T1 The point policy type.
/// @param p The plane.
/// @param pt The point.
/// @return Signed distance from point to plane.
template <std::size_t N, typename T0, typename T1>
auto distance(const plane_like<N, T0> &p, const point_like<N, T1> &pt) {
  return tf::dot(p.normal, pt) + p.d;
}

/// @copydoc distance(const plane_like<N,T0>&,const point_like<N,T1>&)
template <std::size_t N, typename T0, typename T1>
auto distance(const point_like<N, T0> &pt, const plane_like<N, T1> &p) {
  return distance(p, pt);
}

/// @ingroup core_queries
/// @brief Computes the squared distance from a point to a plane.
///
/// @tparam N The dimensionality.
/// @tparam T0 The plane policy type.
/// @tparam T1 The point policy type.
/// @param p The plane.
/// @param pt The point.
/// @return Squared distance from point to plane.
template <std::size_t N, typename T0, typename T1>
auto distance2(const plane_like<N, T0> &p, const point_like<N, T1> &pt) {
  auto d = distance(p, pt);
  return d * d;
}

/// @copydoc distance2(const plane_like<N,T0>&,const point_like<N,T1>&)
template <std::size_t N, typename T0, typename T1>
auto distance2(const point_like<N, T0> &pt, const plane_like<N, T1> &p) {
  return distance2(p, pt);
}

// Segment to Plane
template <std::size_t Dims, typename T0, typename T1>
auto distance2(const segment<Dims, T0> &s, const plane_like<Dims, T1> &p) {
  return closest_metric_point(s, p).metric;
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const segment<Dims, T0> &s, const plane_like<Dims, T1> &p) {
  return tf::sqrt(distance2(s, p));
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const plane_like<Dims, T0> &p, const segment<Dims, T1> &s) {
  return distance2(s, p);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const plane_like<Dims, T0> &p, const segment<Dims, T1> &s) {
  return distance(s, p);
}

// Ray to Plane
template <std::size_t Dims, typename T0, typename T1>
auto distance2(const ray_like<Dims, T0> &r, const plane_like<Dims, T1> &p) {
  return closest_metric_point(r, p).metric;
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const ray_like<Dims, T0> &r, const plane_like<Dims, T1> &p) {
  return tf::sqrt(distance2(r, p));
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const plane_like<Dims, T0> &p, const ray_like<Dims, T1> &r) {
  return distance2(r, p);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const plane_like<Dims, T0> &p, const ray_like<Dims, T1> &r) {
  return distance(r, p);
}

// Line to Plane
template <std::size_t Dims, typename T0, typename T1>
auto distance2(const line_like<Dims, T0> &l, const plane_like<Dims, T1> &p) {
  return closest_metric_point(l, p).metric;
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const line_like<Dims, T0> &l, const plane_like<Dims, T1> &p) {
  return tf::sqrt(distance2(l, p));
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const plane_like<Dims, T0> &p, const line_like<Dims, T1> &l) {
  return distance2(l, p);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const plane_like<Dims, T0> &p, const line_like<Dims, T1> &l) {
  return distance(l, p);
}

// Polygon to Plane
template <std::size_t Dims, typename T0, typename T1>
auto distance2(const polygon<Dims, T0> &poly, const plane_like<Dims, T1> &p) {
  return closest_metric_point(poly, p).metric;
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const polygon<Dims, T0> &poly, const plane_like<Dims, T1> &p) {
  return tf::sqrt(distance2(poly, p));
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const plane_like<Dims, T0> &p, const polygon<Dims, T1> &poly) {
  return distance2(poly, p);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const plane_like<Dims, T0> &p, const polygon<Dims, T1> &poly) {
  return distance(poly, p);
}

// Plane to Plane
template <std::size_t Dims, typename T0, typename T1>
auto distance2(const plane_like<Dims, T0> &p0, const plane_like<Dims, T1> &p1) {
  return closest_metric_point(p0, p1).metric;
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const plane_like<Dims, T0> &p0, const plane_like<Dims, T1> &p1) {
  return tf::sqrt(distance2(p0, p1));
}

// AABB to Plane
template <std::size_t Dims, typename T0, typename T1>
auto distance(const aabb_like<Dims, T0> &aabb, const plane_like<Dims, T1> &p) {
  using T = tf::coordinate_type<T0, T1>;
  auto center = aabb.center();
  auto half_extent = (aabb.max - aabb.min) * T(0.5);

  // Distance from center to plane
  auto d_center = tf::dot(p.normal, center) + p.d;

  // Projected radius: sum of half-extents projected onto plane normal
  T r = T(0);
  for (std::size_t i = 0; i < Dims; ++i) {
    r += half_extent[i] * std::abs(p.normal[i]);
  }

  return std::max(T(0), std::abs(d_center) - r);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const aabb_like<Dims, T0> &aabb, const plane_like<Dims, T1> &p) {
  auto d = distance(aabb, p);
  return d * d;
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const plane_like<Dims, T0> &p, const aabb_like<Dims, T1> &aabb) {
  return distance(aabb, p);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const plane_like<Dims, T0> &p, const aabb_like<Dims, T1> &aabb) {
  return distance2(aabb, p);
}

// OBB to Plane
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::obb_like<Dims, Policy0> &obb,
              const tf::plane_like<Dims, Policy1> &p) {
  using T = tf::coordinate_type<Policy0, Policy1>;

  auto center = obb.origin;
  for (std::size_t i = 0; i < Dims; ++i) {
    center = center + obb.axes[i] * (obb.extent[i] * T(0.5));
  }

  auto d_center = tf::dot(p.normal, center) + p.d;

  T r = T(0);
  for (std::size_t i = 0; i < Dims; ++i) {
    auto proj = tf::dot(p.normal, obb.axes[i]);
    r += std::abs(proj) * (obb.extent[i] * T(0.5));
  }

  return std::max(T(0), std::abs(d_center) - r);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::obb_like<Dims, Policy0> &obb,
               const tf::plane_like<Dims, Policy1> &p) {
  auto d = distance(obb, p);
  return d * d;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::plane_like<Dims, Policy0> &p,
              const tf::obb_like<Dims, Policy1> &obb) {
  return distance(obb, p);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::plane_like<Dims, Policy0> &p,
               const tf::obb_like<Dims, Policy1> &obb) {
  return distance2(obb, p);
}

// RSS to Plane
template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::rss_like<Dims, Policy0> &rss,
              const tf::plane_like<Dims, Policy1> &p) {
  using T = tf::coordinate_type<Policy0, Policy1>;

  static_assert(Dims >= 2, "rss must have at least a 1D rectangle");

  auto center = rss.origin;
  for (std::size_t i = 0; i < Dims - 1; ++i) {
    center = center + rss.axes[i] * (rss.length[i] * T(0.5));
  }

  auto d_center = tf::dot(p.normal, center) + p.d;

  T rect_r = T(0);
  for (std::size_t i = 0; i < Dims - 1; ++i) {
    auto proj = tf::dot(p.normal, rss.axes[i]);
    rect_r += std::abs(proj) * (rss.length[i] * T(0.5));
  }

  T r_total = rect_r + rss.radius;

  return std::max(T(0), std::abs(d_center) - r_total);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::rss_like<Dims, Policy0> &rss,
               const tf::plane_like<Dims, Policy1> &p) {
  auto d = distance(rss, p);
  return d * d;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::plane_like<Dims, Policy0> &p,
              const tf::rss_like<Dims, Policy1> &rss) {
  return distance(rss, p);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::plane_like<Dims, Policy0> &p,
               const tf::rss_like<Dims, Policy1> &rss) {
  return distance2(rss, p);
}

template <std::size_t Dims, typename Policy, typename T1>
auto distance2(const tf::line_like<Dims, Policy> &l,
               const tf::point_like<Dims, T1> &v1) {
  return closest_metric_point(l, v1).metric;
}

template <std::size_t Dims, typename Policy, typename T1>
auto distance(const tf::line_like<Dims, Policy> &l,
              const tf::point_like<Dims, T1> &v1) {
  return tf::sqrt(distance2(l, v1));
}

template <std::size_t Dims, typename T1, typename Policy>
auto distance2(const tf::point_like<Dims, T1> &v0,
               const tf::line_like<Dims, Policy> &l) {
  return closest_metric_point(v0, l).metric;
}

template <std::size_t Dims, typename T1, typename Policy>
auto distance(const tf::point_like<Dims, T1> &v0,
              const tf::line_like<Dims, Policy> &l) {
  return tf::sqrt(distance2(v0, l));
}

template <std::size_t Dims, typename Policy, typename T1>
auto distance2(const tf::ray_like<Dims, Policy> &r,
               const tf::point_like<Dims, T1> &v1) {
  return closest_metric_point(r, v1).metric;
}

template <std::size_t Dims, typename Policy, typename T1>
auto distance(const tf::ray_like<Dims, Policy> &r,
              const tf::point_like<Dims, T1> &v1) {
  return tf::sqrt(distance2(r, v1));
}

template <std::size_t Dims, typename T1, typename Policy>
auto distance2(const tf::point_like<Dims, T1> &v0,
               const tf::ray_like<Dims, Policy> &r) {
  return closest_metric_point(v0, r).metric;
}

template <std::size_t Dims, typename T1, typename Policy>
auto distance(const tf::point_like<Dims, T1> &v0,
              const tf::ray_like<Dims, Policy> &r) {
  return tf::sqrt(distance2(v0, r));
}

template <typename T0, std::size_t Dims, typename T1>
auto distance2(const tf::segment<Dims, T0> &s,
               const tf::point_like<Dims, T1> &v1) {
  return closest_metric_point(s, v1).metric;
}

template <typename T0, std::size_t Dims, typename T1>
auto distance(const tf::segment<Dims, T0> &s,
              const tf::point_like<Dims, T1> &v1) {
  return tf::sqrt(distance2(s, v1));
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const tf::point_like<Dims, T0> &v0,
               const tf::segment<Dims, T1> &s) {
  return closest_metric_point(v0, s).metric;
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const tf::point_like<Dims, T0> &v0,
              const tf::segment<Dims, T1> &s) {
  return tf::sqrt(distance2(v0, s));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::line_like<Dims, Policy0> &l0,
               const tf::line_like<Dims, Policy1> &l1) {
  return closest_metric_point(l0, l1).metric;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::line_like<Dims, Policy0> &l0,
              const tf::line_like<Dims, Policy1> &l1) {
  return tf::sqrt(distance2(l0, l1));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::ray_like<Dims, Policy0> &r0,
               const tf::ray_like<Dims, Policy1> &r1) {
  return closest_metric_point(r0, r1).metric;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::ray_like<Dims, Policy0> &r0,
              const tf::ray_like<Dims, Policy1> &r1) {
  return tf::sqrt(distance2(r0, r1));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::line_like<Dims, Policy0> &l0,
               const tf::ray_like<Dims, Policy1> &r1) {
  return closest_metric_point(l0, r1).metric;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::line_like<Dims, Policy0> &l0,
              const tf::ray_like<Dims, Policy1> &r1) {
  return tf::sqrt(distance2(l0, r1));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::ray_like<Dims, Policy0> &r0,
               const tf::line_like<Dims, Policy1> &l1) {
  return closest_metric_point(r0, l1).metric;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::ray_like<Dims, Policy0> &r0,
              const tf::line_like<Dims, Policy1> &l1) {
  return tf::sqrt(distance2(r0, l1));
}

template <std::size_t Dims, typename Policy, typename T>
auto distance2(const tf::ray_like<Dims, Policy> &r0,
               const tf::segment<Dims, T> &s1) {
  return closest_metric_point(r0, s1).metric;
}

template <std::size_t Dims, typename Policy, typename T>
auto distance(const tf::ray_like<Dims, Policy> &r0,
              const tf::segment<Dims, T> &s1) {
  return tf::sqrt(distance2(r0, s1));
}

template <std::size_t Dims, typename Policy, typename T>
auto distance2(const tf::line_like<Dims, Policy> &l0,
               const tf::segment<Dims, T> &s1) {
  return closest_metric_point(l0, s1).metric;
}

template <std::size_t Dims, typename Policy, typename T>
auto distance(const tf::line_like<Dims, Policy> &l0,
              const tf::segment<Dims, T> &s1) {
  return tf::sqrt(distance2(l0, s1));
}

template <typename T, std::size_t Dims, typename Policy>
auto distance2(const tf::segment<Dims, T> &s0,
               const tf::line_like<Dims, Policy> &l1) {
  return closest_metric_point(s0, l1).metric;
}

template <typename T, std::size_t Dims, typename Policy>
auto distance(const tf::segment<Dims, T> &s0,
              const tf::line_like<Dims, Policy> &l1) {
  return tf::sqrt(distance2(s0, l1));
}

template <typename T, std::size_t Dims, typename Policy>
auto distance2(const tf::segment<Dims, T> &s0,
               const tf::ray_like<Dims, Policy> &r1) {
  return closest_metric_point(s0, r1).metric;
}

template <typename T, std::size_t Dims, typename Policy>
auto distance(const tf::segment<Dims, T> &s0,
              const tf::ray_like<Dims, Policy> &r1) {
  return tf::sqrt(distance2(s0, r1));
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const tf::segment<Dims, T0> &s0,
               const tf::segment<Dims, T1> &s1) {
  return closest_metric_point(s0, s1).metric;
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const tf::segment<Dims, T0> &s0,
              const tf::segment<Dims, T1> &s1) {
  return tf::sqrt(distance2(s0, s1));
}

template <typename Policy0, std::size_t Dims, typename Policy1>
auto distance2(const tf::polygon<Dims, Policy0> &poly_in,
               const tf::point_like<Dims, Policy1> &pt) {
  return closest_metric_point(poly_in, pt).metric;
}

template <typename Policy0, std::size_t Dims, typename Policy1>
auto distance(const tf::polygon<Dims, Policy0> &poly_in,
              const tf::point_like<Dims, Policy1> &pt) {
  return tf::sqrt(distance2(poly_in, pt));
}

template <std::size_t Dims, typename Policy1, typename Policy0>
auto distance2(const tf::point_like<Dims, Policy1> &pt,
               const tf::polygon<Dims, Policy0> &poly) {
  return closest_metric_point(pt, poly).metric;
}

template <std::size_t Dims, typename Policy1, typename Policy0>
auto distance(const tf::point_like<Dims, Policy1> &pt,
              const tf::polygon<Dims, Policy0> &poly) {
  return tf::sqrt(distance2(pt, poly));
}

template <typename Policy0, std::size_t Dims, typename Policy>
auto distance2(const tf::polygon<Dims, Policy0> &poly_in,
               const tf::line_like<Dims, Policy> &line) {
  return closest_metric_point(poly_in, line).metric;
}

template <typename Policy0, std::size_t Dims, typename Policy>
auto distance(const tf::polygon<Dims, Policy0> &poly_in,
              const tf::line_like<Dims, Policy> &line) {
  return tf::sqrt(distance2(poly_in, line));
}

template <std::size_t Dims, typename Policy, typename Policy0>
auto distance2(const tf::line_like<Dims, Policy> &line,
               const tf::polygon<Dims, Policy0> &poly) {
  return closest_metric_point(line, poly).metric;
}

template <std::size_t Dims, typename Policy, typename Policy0>
auto distance(const tf::line_like<Dims, Policy> &line,
              const tf::polygon<Dims, Policy0> &poly) {
  return tf::sqrt(distance2(line, poly));
}

template <typename Policy0, std::size_t Dims, typename Policy>
auto distance2(const tf::polygon<Dims, Policy0> &poly_in,
               const tf::ray_like<Dims, Policy> &ray) {
  return closest_metric_point(poly_in, ray).metric;
}

template <typename Policy0, std::size_t Dims, typename Policy>
auto distance(const tf::polygon<Dims, Policy0> &poly_in,
              const tf::ray_like<Dims, Policy> &ray) {
  return tf::sqrt(distance2(poly_in, ray));
}

template <std::size_t Dims, typename Policy, typename Policy0>
auto distance2(const tf::ray_like<Dims, Policy> &ray,
               const tf::polygon<Dims, Policy0> &poly) {
  return closest_metric_point(ray, poly).metric;
}

template <std::size_t Dims, typename Policy, typename Policy0>
auto distance(const tf::ray_like<Dims, Policy> &ray,
              const tf::polygon<Dims, Policy0> &poly) {
  return tf::sqrt(distance2(ray, poly));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::polygon<Dims, Policy0> &poly_in,
               const tf::segment<Dims, Policy1> &seg1) {
  return closest_metric_point(poly_in, seg1).metric;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::polygon<Dims, Policy0> &poly_in,
              const tf::segment<Dims, Policy1> &seg1) {
  return tf::sqrt(distance2(poly_in, seg1));
}

template <typename Policy, std::size_t Dims, typename Policy0>
auto distance2(const tf::segment<Dims, Policy> &seg,
               const tf::polygon<Dims, Policy0> &poly) {
  return closest_metric_point(seg, poly).metric;
}

template <typename Policy, std::size_t Dims, typename Policy0>
auto distance(const tf::segment<Dims, Policy> &seg,
              const tf::polygon<Dims, Policy0> &poly) {
  return tf::sqrt(distance2(seg, poly));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::polygon<Dims, Policy0> &poly_in0,
               const tf::polygon<Dims, Policy1> &poly_in1) {
  return closest_metric_point(poly_in0, poly_in1).metric;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::polygon<Dims, Policy0> &poly_in0,
              const tf::polygon<Dims, Policy1> &poly_in1) {
  return tf::sqrt(distance2(poly_in0, poly_in1));
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const sphere_like<Dims, T0> &s, const point_like<Dims, T1> &pt) {
  auto d2 = (s.origin - pt).length2();
  if (d2 + tf::epsilon2<decltype(d2)> < s.r * s.r)
    return decltype(d2)(0);
  return tf::sqrt(d2) - s.r;
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const point_like<Dims, T1> &pt, const sphere_like<Dims, T0> &s) {
  return distance(pt, s);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const sphere_like<Dims, T0> &s, const point_like<Dims, T1> &pt) {
  auto d = distance(s, pt);
  return d * d;
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const point_like<Dims, T1> &pt, const sphere_like<Dims, T0> &s) {
  return distance2(pt, s);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const sphere_like<Dims, T0> &s, const ray_like<Dims, T1> &r) {
  auto t = closest_point_parametric(r, s.origin);
  return distance(s, r.origin + t * r.direction);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const ray_like<Dims, T1> &r, const sphere_like<Dims, T0> &s) {
  return distance(r, s);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const sphere_like<Dims, T0> &s, const ray_like<Dims, T1> &r) {
  auto d = distance(s, r);
  return d * d;
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const ray_like<Dims, T1> &r, const sphere_like<Dims, T0> &s) {
  return distance2(r, s);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const sphere_like<Dims, T0> &s, const line_like<Dims, T1> &l) {
  auto t = closest_point_parametric(l, s.origin);
  return distance(s, l.origin + t * l.direction);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const line_like<Dims, T1> &l, const sphere_like<Dims, T0> &s) {
  return distance(l, s);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const sphere_like<Dims, T0> &s, const line_like<Dims, T1> &l) {
  auto d = distance(s, l);
  return d * d;
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const line_like<Dims, T1> &l, const sphere_like<Dims, T0> &s) {
  return distance2(l, s);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const sphere_like<Dims, T0> &s, const segment<Dims, T1> &seg) {
  auto t = closest_point_parametric(seg, s.origin);
  return distance(s, seg[0] + t * (seg[1] - seg[0]));
}

template <std::size_t Dims, typename T0, typename T1>
auto distance(const segment<Dims, T1> &seg, const sphere_like<Dims, T0> &s) {
  return distance(seg, s);
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const sphere_like<Dims, T0> &s, const segment<Dims, T1> &seg) {
  auto d = distance(s, seg);
  return d * d;
}

template <std::size_t Dims, typename T0, typename T1>
auto distance2(const segment<Dims, T1> &seg, const sphere_like<Dims, T0> &s) {
  return distance2(seg, s);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::rss_like<Dims, Policy0> &rss0,
              const tf::rss_like<Dims, Policy1> &rss1) {
  static_assert(Dims == 2 || Dims == 3,
                "RSS distance is implemented for 2D and 3D only.");
  using T = tf::coordinate_type<Policy0, Policy1>;

  // Rotation matrix: Rab = A^T * B
  std::array<std::array<T, Dims>, Dims> rot_ab;
  for (std::size_t i = 0; i < Dims; ++i) {
    for (std::size_t j = 0; j < Dims; ++j) {
      rot_ab[i][j] = tf::dot(rss0.axes[i], rss1.axes[j]);
    }
  }

  // Translation Tab = origin1 - origin0, expressed in frame A
  auto diff = rss1.origin - rss0.origin;
  std::array<T, Dims> tr_ab;
  for (std::size_t i = 0; i < Dims; ++i) {
    tr_ab[i] = tf::dot(diff, rss0.axes[i]);
  }

  T base_dist;
  if constexpr (Dims == 2) {
    // 2D: base shape is a segment (length[0])
    base_dist = tf::core::local_segment_distance(rot_ab, tr_ab,
                                                  rss0.length[0], rss1.length[0]);
  } else {
    // 3D: base shape is a rectangle (length[0] x length[1])
    std::array<T, 2> a{rss0.length[0], rss0.length[1]};
    std::array<T, 2> b{rss1.length[0], rss1.length[1]};
    base_dist = tf::core::local_rectangle_distance(rot_ab, tr_ab, a, b);
  }

  T rss_dist = base_dist - rss0.radius - rss1.radius;
  return std::max(rss_dist, T(0));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::rss_like<Dims, Policy0> &rss0,
               const tf::rss_like<Dims, Policy1> &rss1) {
  static_assert(Dims == 2 || Dims == 3,
                "RSS distance is implemented for 2D and 3D only.");
  auto out = distance(rss0, rss1);
  return out * out;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::rss_like<Dims, Policy0> &rss,
              const tf::point_like<Dims, Policy1> &pt) {
  static_assert(Dims == 2 || Dims == 3,
                "distance(rss, point) is implemented for 2D and 3D only.");
  using T = tf::coordinate_type<Policy0, Policy1>;

  // Transform point to local coordinates
  auto diff = pt - rss.origin;
  std::array<T, Dims> local_pt;
  for (std::size_t i = 0; i < Dims; ++i) {
    local_pt[i] = tf::dot(diff, rss.axes[i]);
  }

  // Distance to base shape in local coords
  T base_dist;
  if constexpr (Dims == 2) {
    // 2D: base shape is a segment (length[0])
    base_dist = tf::core::local_point_segment_distance(local_pt, rss.length[0]);
  } else {
    // 3D: base shape is a rectangle (length[0] x length[1])
    std::array<T, 2> length{rss.length[0], rss.length[1]};
    base_dist = tf::core::local_point_rectangle_distance(local_pt, length);
  }

  // Subtract radius
  T rss_dist = base_dist - rss.radius;
  return std::max(rss_dist, T(0));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::rss_like<Dims, Policy0> &rss,
               const tf::point_like<Dims, Policy1> &pt) {
  auto d = distance(rss, pt);
  return d * d;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::point_like<Dims, Policy0> &pt,
              const tf::rss_like<Dims, Policy1> &rss) {
  return distance(rss, pt);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::point_like<Dims, Policy0> &pt,
               const tf::rss_like<Dims, Policy1> &rss) {
  return distance2(rss, pt);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::obb_like<Dims, Policy0> &obb,
               const tf::point_like<Dims, Policy1> &pt) {
  using T = tf::coordinate_type<Policy0, Policy1>;

  // Transform point to local coordinates
  auto diff = pt - obb.origin;
  std::array<T, Dims> local_pt;
  for (std::size_t i = 0; i < Dims; ++i) {
    local_pt[i] = tf::dot(diff, obb.axes[i]);
  }

  // Distance to box in local coords
  std::array<T, Dims> extent;
  for (std::size_t i = 0; i < Dims; ++i) {
    extent[i] = obb.extent[i];
  }
  return tf::core::local_point_box_distance2(local_pt, extent);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::obb_like<Dims, Policy0> &obb,
              const tf::point_like<Dims, Policy1> &pt) {
  return tf::sqrt(distance2(obb, pt));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::point_like<Dims, Policy0> &pt,
               const tf::obb_like<Dims, Policy1> &obb) {
  return distance2(obb, pt);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::point_like<Dims, Policy0> &pt,
              const tf::obb_like<Dims, Policy1> &obb) {
  return distance(obb, pt);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::obbrss_like<Dims, Policy0> &obbrss0,
              const tf::obbrss_like<Dims, Policy1> &obbrss1) {
  static_assert(Dims == 3,
                "obbrss_metrics is currently implemented for 3D only.");
  using T = tf::coordinate_type<Policy0, Policy1>;

  using std::max;

  // Rotation matrix: Rab = A^T * B
  std::array<std::array<T, 3>, 3> rot_ab;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      rot_ab[i][j] = tf::dot(obbrss0.axes[i], obbrss1.axes[j]);
    }
  }

  // Translation Tab = rss_origin1 - rss_origin0, expressed in frame A
  // Use rss_origin (not obb_origin) for the RSS rectangle distance
  auto diff = obbrss1.rss_origin - obbrss0.rss_origin;
  std::array<T, 3> tr_ab;
  for (int i = 0; i < 3; ++i) {
    tr_ab[i] = tf::dot(diff, obbrss0.axes[i]);
  }

  // Rectangle side lengths
  std::array<T, 2> a{obbrss0.length[0], obbrss0.length[1]};
  std::array<T, 2> b{obbrss1.length[0], obbrss1.length[1]};

  // Distance between the two oriented rectangles
  auto rect_dist = tf::core::local_rectangle_distance(rot_ab, tr_ab, a, b);
  T obbrss_dist = rect_dist - obbrss0.radius - obbrss1.radius;
  if (obbrss_dist < T(0)) {
    obbrss_dist = T(0);
  }
  return obbrss_dist;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::obbrss_like<Dims, Policy0> &obbrss0,
               const tf::obbrss_like<Dims, Policy1> &obbrss1) {
  static_assert(Dims == 3,
                "obbrss_metrics is currently implemented for 3D only.");
  auto out = distance(obbrss0, obbrss1);
  return out * out;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::obbrss_like<Dims, Policy0> &obbrss,
              const tf::point_like<Dims, Policy1> &pt) {
  static_assert(Dims == 3,
                "distance(obbrss, point) is implemented for 3D only.");
  using T = tf::coordinate_type<Policy0, Policy1>;

  // Transform point to local coordinates (using RSS origin)
  auto diff = pt - obbrss.rss_origin;
  std::array<T, 3> local_pt;
  for (int i = 0; i < 3; ++i) {
    local_pt[i] = tf::dot(diff, obbrss.axes[i]);
  }

  // Distance to rectangle in local coords
  std::array<T, 2> length{obbrss.length[0], obbrss.length[1]};
  auto rect_dist = tf::core::local_point_rectangle_distance(local_pt, length);

  // Subtract radius
  T obbrss_dist = rect_dist - obbrss.radius;
  return std::max(obbrss_dist, T(0));
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::obbrss_like<Dims, Policy0> &obbrss,
               const tf::point_like<Dims, Policy1> &pt) {
  auto d = distance(obbrss, pt);
  return d * d;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance(const tf::point_like<Dims, Policy0> &pt,
              const tf::obbrss_like<Dims, Policy1> &obbrss) {
  return distance(obbrss, pt);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto distance2(const tf::point_like<Dims, Policy0> &pt,
               const tf::obbrss_like<Dims, Policy1> &obbrss) {
  return distance2(obbrss, pt);
}

namespace core {
template <typename Obj> struct distance_with {
  Obj obj;
  template <typename T> auto operator()(const T &t) const {
    return tf::distance(obj, t);
  }
};

template <typename Obj> struct distance2_with {
  Obj obj;
  template <typename T> auto operator()(const T &t) const {
    return tf::distance2(obj, t);
  }
};

struct distancer {
  template <typename T> auto operator()(T &&t) const {
    return core::distance_with<std::decay_t<T>>{static_cast<T &&>(t)};
  }

  template <typename T0, typename T1>
  auto operator()(const T0 &t0, const T1 &t1) const {
    return tf::distance(t0, t1);
  }
};

struct distancer2 {
  template <typename T> auto operator()(T &&t) const {
    return core::distance2_with<std::decay_t<T>>{static_cast<T &&>(t)};
  }

  template <typename T0, typename T1>
  auto operator()(const T0 &t0, const T1 &t1) const {
    return tf::distance2(t0, t1);
  }
};
} // namespace core

constexpr core::distancer distance_f;
constexpr core::distancer2 distance2_f;

} // namespace tf

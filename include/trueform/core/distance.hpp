/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "./aabb_like.hpp"
#include "./closest_metric_point.hpp"
#include "./coordinate_type.hpp"
#include "./dot.hpp"
#include "./line_like.hpp"
#include "./plane_like.hpp"
#include "./point_like.hpp"
#include "./ray_like.hpp"
#include "./segment.hpp"
#include "./sphere_like.hpp"
#include "./sqrt.hpp"

namespace tf {

/// @ingroup geometry
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

/// @ingroup geometry
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

/// @ingroup geometry
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

/// @ingroup geometry
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

/// @ingroup geometry
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

/// @ingroup geometry
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

/// @ingroup geometry
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

/// @ingroup geometry
/// @brief Computes the squared distance from a point to an AABB (reverse
/// argument order).
template <std::size_t N, typename T0, typename T1>
auto distance2(const point_like<N, T0> &_point, const aabb_like<N, T1> &_bbox) {
  return distance2(_bbox, _point);
}

/// @ingroup geometry
/// @brief Computes the distance from a point to an AABB.
template <std::size_t N, typename T0, typename T1>
auto distance(const aabb_like<N, T0> &_bbox, const point_like<N, T1> &_point) {
  return tf::sqrt(distance2(_bbox, _point));
}

/// @ingroup geometry
/// @brief Computes the distance from a point to an AABB (reverse argument
/// order).
template <std::size_t N, typename T0, typename T1>
auto distance(const point_like<N, T0> &_point, const aabb_like<N, T1> &_bbox) {
  return tf::sqrt(distance2(_bbox, _point));
}

/// @ingroup geometry
/// @brief Computes the distance from a point to an AABB (reverse argument
/// order).
template <std::size_t N, typename T0, typename T1>
auto distance(const plane_like<N, T0> &p, const point_like<N, T1> &pt) {
  return tf::dot(p.normal, pt) + p.d;
}

/// @ingroup geometry
/// @brief Computes the distance from a point to an AABB (reverse argument
/// order).
template <std::size_t N, typename T0, typename T1>
auto distance(const point_like<N, T0> &pt, const plane_like<N, T1> &p) {
  return distance(p, pt);
}

/// @ingroup geometry
/// @brief Computes the distance from a point to an AABB (reverse argument
/// order).
template <std::size_t N, typename T0, typename T1>
auto distance2(const plane_like<N, T0> &p, const point_like<N, T1> &pt) {
  auto d = distance(p, pt);
  return d * d;
}

/// @ingroup geometry
/// @brief Computes the distance from a point to an AABB (reverse argument
/// order).
template <std::size_t N, typename T0, typename T1>
auto distance2(const point_like<N, T0> &pt, const plane_like<N, T1> &p) {
  return distance2(p, pt);
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

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
#include "./algorithm/min.hpp"
#include "./closest_point_on_triangle.hpp"
#include "./closest_point_parametric.hpp"
#include "./closest_points_on_triangles.hpp"
#include "./contains_coplanar_point.hpp"
#include "./line_like.hpp"
#include "./metric_point_pair.hpp"
#include "./polygon.hpp"
#include "./ray_hit.hpp"

namespace tf {

template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point_pair(const tf::point_like<Dims, T> &pt,
                               const tf::plane_like<Dims, Policy> &p) {
  auto d = tf::dot(p.normal, pt) + p.d;
  return tf::make_metric_point_pair(d * d, pt, pt - d * p.normal);
}

template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point_pair(const tf::plane_like<Dims, T> &p,
                               const tf::point_like<Dims, Policy> &pt) {
  auto res = closest_metric_point_pair(pt, p);
  std::swap(res.first, res.second);
  return res;
}

template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point_pair(const tf::segment<Dims, T> &s0,
                               const tf::plane_like<Dims, Policy> &p1) {
  auto l0 = tf::make_line_between_points(s0[0], s0[1]);
  auto t = tf::closest_point_parametric(s0, p1);
  auto pt0 = l0.origin + t * l0.direction;
  return tf::closest_metric_point_pair(pt0, p1);
}

template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point_pair(const tf::plane_like<Dims, T> &o0,
                               const tf::segment<Dims, Policy> &o1) {
  auto res = closest_metric_point_pair(o1, o0);
  std::swap(res.first, res.second);
  return res;
}

template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point_pair(const tf::ray_like<Dims, T> &o0,
                               const tf::plane_like<Dims, Policy> &p1) {
  auto t = tf::closest_point_parametric(o0, p1);
  auto pt0 = o0.origin + t * o0.direction;
  return tf::closest_metric_point_pair(pt0, p1);
}

template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point_pair(const tf::plane_like<Dims, T> &o0,
                               const tf::ray_like<Dims, Policy> &o1) {
  auto res = closest_metric_point_pair(o1, o0);
  std::swap(res.first, res.second);
  return res;
}

template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point_pair(const tf::line_like<Dims, T> &o0,
                               const tf::plane_like<Dims, Policy> &p1) {
  auto t = tf::closest_point_parametric(o0, p1);
  auto pt0 = o0.origin + t * o0.direction;
  return tf::closest_metric_point_pair(pt0, p1);
}

template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point_pair(const tf::plane_like<Dims, T> &o0,
                               const tf::line_like<Dims, Policy> &o1) {
  auto res = closest_metric_point_pair(o1, o0);
  std::swap(res.first, res.second);
  return res;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point_pair(const tf::plane_like<Dims, Policy0> &p0,
                               const tf::plane_like<Dims, Policy1> &p1) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  auto dot_n = tf::dot(p0.normal, p1.normal);

  if (std::abs(dot_n) < T(1) - tf::epsilon<T>) {
    auto dir = tf::cross(p0.normal, p1.normal);
    auto n0xn1 = dir.length2();
    auto pt = tf::make_point(
        (p1.d * dot_n - p0.d) / n0xn1 * tf::cross(dir, p1.normal) +
        (p0.d * dot_n - p1.d) / n0xn1 * tf::cross(p0.normal, dir));
    return tf::make_metric_point_pair(T(0), pt, pt);
  }

  T d_diff = p1.d - p0.d * dot_n;
  if (std::abs(d_diff) < tf::epsilon<T>) {
    tf::point<T, Dims> pt = tf::make_point(-p0.d * p0.normal);
    return tf::make_metric_point_pair(T(0), pt, pt);
  }

  tf::point<T, Dims> pt0 = tf::make_point(-p0.d * p0.normal);
  tf::point<T, Dims> pt1 = pt0 - d_diff * p1.normal;
  return tf::make_metric_point_pair(d_diff * d_diff, pt0, pt1);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point_pair(const tf::polygon<Dims, Policy0> &poly,
                               const tf::plane_like<Dims, Policy1> &plane) {
  using T = tf::coordinate_type<Policy0, Policy1>;
  auto best =
      tf::make_metric_point_pair(std::numeric_limits<T>::max(),
                                 tf::point<T, Dims>{}, tf::point<T, Dims>{});
  std::size_t size = poly.size();
  std::size_t prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++) {
    auto seg = tf::make_segment_between_points(poly[prev], poly[i]);
    auto tmp = tf::closest_metric_point_pair(seg, plane);
    if (tmp.metric < best.metric)
      best = tmp;
    if (best.metric < tf::epsilon2<T>)
      return best;
  }
  return best;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point_pair(const tf::plane_like<Dims, Policy0> &plane,
                               const tf::polygon<Dims, Policy1> &poly) {
  auto res = closest_metric_point_pair(poly, plane);
  std::swap(res.first, res.second);
  return res;
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename T0, typename T1>
auto closest_metric_point_pair(const tf::point_like<Dims, T0> &v0,
                               const tf::point_like<Dims, T1> &v1) {
  return tf::make_metric_point_pair((v0 - v1).length2(), v0, v1);
}
/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename Policy, typename T1>
auto closest_metric_point_pair(const tf::line_like<Dims, Policy> &l,
                               const tf::point_like<Dims, T1> &v1) {
  auto t = tf::closest_point_parametric(l, v1);
  auto pt = l.origin + t * l.direction;
  return tf::make_metric_point_pair((pt - v1).length2(), pt, v1);
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename T1, typename Policy1>
auto closest_metric_point_pair(const tf::point_like<Dims, T1> &v0,
                               const tf::line_like<Dims, Policy1> &l) {
  auto t = tf::closest_point_parametric(l, v0);
  auto pt = l.origin + t * l.direction;
  return tf::make_metric_point_pair((pt - v0).length2(), v0, pt);
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename Policy, typename T1>
auto closest_metric_point_pair(const tf::ray_like<Dims, Policy> &r,
                               const tf::point_like<Dims, T1> &v1) {
  auto t = tf::closest_point_parametric(r, v1);
  auto pt = r.origin + t * r.direction;
  return tf::make_metric_point_pair((pt - v1).length2(), pt, v1);
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename T1, typename Policy>
auto closest_metric_point_pair(const tf::point_like<Dims, T1> &v0,
                               const tf::ray_like<Dims, Policy> &r) {
  auto t = tf::closest_point_parametric(r, v0);
  auto pt = r.origin + t * r.direction;
  return tf::make_metric_point_pair((pt - v0).length2(), v0, pt);
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename T0, typename T1>
auto closest_metric_point_pair(const tf::segment<Dims, T0> &s,
                               const tf::point_like<Dims, T1> &v1) {
  auto t = tf::closest_point_parametric(s, v1);
  auto l = tf::make_line_between_points(s[0], s[1]);
  auto pt = l.origin + t * l.direction;
  return tf::make_metric_point_pair((pt - v1).length2(), pt, v1);
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename T0, typename T1>
auto closest_metric_point_pair(const tf::point_like<Dims, T0> &v0,
                               const tf::segment<Dims, T1> &s) {
  auto t = tf::closest_point_parametric(s, v0);
  auto l = tf::make_line_between_points(s[0], s[1]);
  auto pt = l.origin + t * l.direction;
  return tf::make_metric_point_pair((pt - v0).length2(), v0, pt);
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point_pair(const tf::line_like<Dims, Policy0> &l0,
                               const tf::line_like<Dims, Policy1> &l1) {
  auto [t0, t1] = tf::closest_point_parametric(l0, l1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  return tf::make_metric_point_pair((pt0 - pt1).length2(), pt0, pt1);
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point_pair(const tf::ray_like<Dims, Policy0> &r0,
                               const tf::ray_like<Dims, Policy1> &r1) {
  auto [t0, t1] = tf::closest_point_parametric(r0, r1);
  auto pt0 = r0.origin + t0 * r0.direction;
  auto pt1 = r1.origin + t1 * r1.direction;
  return tf::make_metric_point_pair((pt0 - pt1).length2(), pt0, pt1);
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point_pair(const tf::line_like<Dims, Policy0> &l0,
                               const tf::ray_like<Dims, Policy1> &r1) {
  auto [t0, t1] = tf::closest_point_parametric(l0, r1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = r1.origin + t1 * r1.direction;
  return tf::make_metric_point_pair((pt0 - pt1).length2(), pt0, pt1);
}
/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point_pair(const tf::ray_like<Dims, Policy0> &r0,
                               const tf::line_like<Dims, Policy1> &l1) {
  auto [t0, t1] = tf::closest_point_parametric(r0, l1);
  auto pt0 = r0.origin + t0 * r0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  return tf::make_metric_point_pair((pt0 - pt1).length2(), pt0, pt1);
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename Policy, typename T>
auto closest_metric_point_pair(const tf::ray_like<Dims, Policy> &r0,
                               const tf::segment<Dims, T> &s1) {
  auto l1 = tf::make_line_between_points(s1[0], s1[1]);
  auto [t0, t1] = tf::closest_point_parametric(r0, s1);
  auto pt0 = r0.origin + t0 * r0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  return tf::make_metric_point_pair((pt0 - pt1).length2(), pt0, pt1);
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename Policy, typename T>
auto closest_metric_point_pair(const tf::line_like<Dims, Policy> &l0,
                               const tf::segment<Dims, T> &s1) {
  auto l1 = tf::make_line_between_points(s1[0], s1[1]);
  auto [t0, t1] = tf::closest_point_parametric(l0, s1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  return tf::make_metric_point_pair((pt0 - pt1).length2(), pt0, pt1);
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point_pair(const tf::segment<Dims, T> &s0,
                               const tf::line_like<Dims, Policy> &l1) {
  auto l0 = tf::make_line_between_points(s0[0], s0[1]);
  auto [t0, t1] = tf::closest_point_parametric(s0, l1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  return tf::make_metric_point_pair((pt0 - pt1).length2(), pt0, pt1);
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <typename T, std::size_t Dims, typename Policy>
auto closest_metric_point_pair(const tf::segment<Dims, T> &s0,
                               const tf::ray_like<Dims, Policy> &r1) {
  auto l0 = tf::make_line_between_points(s0[0], s0[1]);
  auto [t0, t1] = tf::closest_point_parametric(s0, r1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = r1.origin + t1 * r1.direction;
  return tf::make_metric_point_pair((pt0 - pt1).length2(), pt0, pt1);
}

template <std::size_t Dims, typename T0, typename T1>
auto closest_metric_point_pair(const tf::segment<Dims, T0> &s0,
                               const tf::segment<Dims, T1> &s1) {
  auto l0 = tf::make_line_between_points(s0[0], s0[1]);
  auto l1 = tf::make_line_between_points(s1[0], s1[1]);
  auto [t0, t1] = tf::closest_point_parametric(s0, s1);
  auto pt0 = l0.origin + t0 * l0.direction;
  auto pt1 = l1.origin + t1 * l1.direction;
  return tf::make_metric_point_pair((pt0 - pt1).length2(), pt0, pt1);
}

namespace core {
template <typename Policy0, typename Policy1>
auto closest_metric_point_pair2d(const tf::polygon<2, Policy0> &poly,
                                 const tf::point_like<2, Policy1> &pt) {
  if constexpr (tf::static_size_v<Policy0> == 3) {
    auto c_pt = tf::closest_point_on_triangle(poly, pt);
    return tf::make_metric_point_pair((c_pt - pt).length2(), c_pt, pt);
  } else {
    if (poly.size() == 3) {
      auto c_pt = tf::closest_point_on_triangle(poly, pt);
      return tf::make_metric_point_pair((c_pt - pt).length2(), c_pt, pt);
    } else {
      tf::coordinate_type<Policy0, Policy1> d2 =
          std::numeric_limits<tf::coordinate_type<Policy0, Policy1>>::max();
      tf::point<decltype(d2), 2> c_pt = pt;
      auto res = tf::make_metric_point_pair(decltype(d2)(0), c_pt, pt);
      if (tf::contains_coplanar_point(poly, pt))
        return res;
      res.metric = d2;
      std::size_t size = poly.size();
      std::size_t prev = size - 1;
      for (std::size_t i = 0; i < size; prev = i++) {
        res = min(
            res, closest_metric_point_pair(
                     tf::make_segment_between_points(poly[prev], poly[i]), pt));
      }
      return res;
    }
  }
}
} // namespace core

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <typename Policy0, std::size_t Dims, typename Policy1>
auto closest_metric_point_pair(const tf::polygon<Dims, Policy0> &poly_in,
                               const tf::point_like<Dims, Policy1> &pt) {
  if constexpr (Dims == 2) {
    return core::closest_metric_point_pair2d(poly_in, pt);
  } else {
    if constexpr (tf::static_size_v<Policy0> == 3) {
      auto c_pt = tf::closest_point_on_triangle(poly_in, pt);
      return tf::make_metric_point_pair((c_pt - pt).length2(), c_pt, pt);
    } else {
      if (poly_in.size() == 3) {
        auto c_pt = tf::closest_point_on_triangle(poly_in, pt);
        return tf::make_metric_point_pair((c_pt - pt).length2(), c_pt, pt);
      } else {
        const auto &poly = tf::tag_plane(poly_in);
        auto d = tf::dot(poly.plane().normal, pt) + poly.plane().d;
        auto c_pt = pt - d * poly.plane().normal;
        auto res = tf::make_metric_point_pair(d * d, c_pt, pt);
        if (tf::contains_coplanar_point(poly, c_pt)) {
          if (std::abs(d) < tf::epsilon<decltype(d)>) {
            res.metric = 0;
            return res;
          }
        } else
          res.metric = std::numeric_limits<decltype(d)>::max();
        std::size_t size = poly.size();
        std::size_t prev = size - 1;
        for (std::size_t i = 0; i < size; prev = i++) {
          res =
              min(res, closest_metric_point_pair(
                           tf::make_segment_between_points(poly[prev], poly[i]),
                           pt));
        }
        return res;
      }
    }
  }
}

template <std::size_t Dims, typename Policy1, typename Policy0>
auto closest_metric_point_pair(const tf::point_like<Dims, Policy1> &pt,
                               const tf::polygon<Dims, Policy0> &poly) {
  auto res = closest_metric_point_pair(poly, pt);
  std::swap(res.first, res.second);
  return res;
}

namespace core {
template <std::size_t Dims, typename Policy0, typename Policy>
auto closest_metric_point_pair_impl(const tf::polygon<Dims, Policy0> &poly,
                                    const tf::line_like<Dims, Policy> &line) {
  using RealT = tf::coordinate_type<Policy0, Policy>;
  auto hit_info =
      tf::ray_hit(tf::make_ray(line.origin, line.direction), poly,
                  tf::make_ray_config(-std::numeric_limits<RealT>::max(),
                                      std::numeric_limits<RealT>::max()));
  if (hit_info) {
    return tf::make_metric_point_pair(RealT(0), hit_info.point, hit_info.point);
  }
  auto best = tf::make_metric_point_pair(std::numeric_limits<RealT>::max(),
                                         hit_info.point, hit_info.point);
  std::size_t size = poly.size();
  std::size_t prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++) {
    auto seg = tf::make_segment_between_points(poly[prev], poly[i]);
    auto tmp = tf::closest_metric_point_pair(seg, line);
    if (tmp.metric < best.metric)
      best = tmp;
  }
  return best;
}
} // namespace core

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename Policy0, typename Policy>
auto closest_metric_point_pair(const tf::polygon<Dims, Policy0> &poly_in,
                               const tf::line_like<Dims, Policy> &line) {
  if constexpr (Dims == 2) {
    return core::closest_metric_point_pair_impl(poly_in, line);
  } else {
    const auto &poly = tf::tag_plane(poly_in);
    return core::closest_metric_point_pair_impl(poly, line);
  }
}

template <std::size_t Dims, typename Policy, typename Policy0>
auto closest_metric_point_pair(const tf::line_like<Dims, Policy> &line,
                               const tf::polygon<Dims, Policy0> &poly) {
  auto res = closest_metric_point_pair(poly, line);
  std::swap(res.first, res.second);
  return res;
}

namespace core {
template <std::size_t Dims, typename Policy0, typename Policy>
auto closest_metric_point_pair_impl(const tf::polygon<Dims, Policy0> &poly,
                                    const tf::ray_like<Dims, Policy> &ray) {
  using RealT = tf::coordinate_type<Policy0, Policy>;
  auto hit_info = tf::ray_hit(ray, poly);
  if (hit_info) {
    return tf::make_metric_point_pair(RealT(0), hit_info.point, hit_info.point);
  }
  auto best = tf::closest_metric_point_pair(poly, ray.origin);
  std::size_t size = poly.size();
  std::size_t prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++) {
    auto seg = tf::make_segment_between_points(poly[prev], poly[i]);
    auto tmp = tf::closest_metric_point_pair(seg, ray);
    if (tmp.metric < best.metric)
      best = tmp;
  }
  return best;
}
} // namespace core

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename Policy0, typename Policy>
auto closest_metric_point_pair(const tf::polygon<Dims, Policy0> &poly_in,
                               const tf::ray_like<Dims, Policy> &ray) {
  if constexpr (Dims == 2) {
    return core::closest_metric_point_pair_impl(poly_in, ray);
  } else {
    const auto &poly = tf::tag_plane(poly_in);
    return core::closest_metric_point_pair_impl(poly, ray);
  }
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename Policy, typename Policy0>
auto closest_metric_point_pair(const tf::ray_like<Dims, Policy> &ray,
                               const tf::polygon<Dims, Policy0> &poly) {
  auto res = closest_metric_point_pair(poly, ray);
  std::swap(res.first, res.second);
  return res;
}

namespace core {
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point_pair_impl(const tf::polygon<Dims, Policy0> &poly,
                                    const tf::segment<Dims, Policy1> &seg1) {
  auto ray = tf::make_ray_between_points(seg1[0], seg1[1]);
  using RealT = tf::coordinate_type<decltype(poly[0][0]), decltype(seg1[0][0])>;
  auto hit_info =
      tf::ray_hit(ray, poly, tf::make_ray_config(RealT(0), RealT(1)));
  if (hit_info) {
    return tf::make_metric_point_pair(RealT(0), hit_info.point, hit_info.point);
  }
  // only check the first vertex
  auto best = closest_metric_point_pair(poly, seg1[0]);
  std::size_t size = poly.size();
  std::size_t prev = size - 1;
  for (std::size_t i = 1; i < size; prev = i++) {
    auto seg = tf::make_segment_between_points(poly[prev], poly[i]);
    best = min(best, closest_metric_point_pair(seg, seg1));
  }
  return best;
}

template <typename Policy, std::size_t Dims, typename Policy0>
auto closest_metric_point_pair_impl(const tf::segment<Dims, Policy> &seg,
                                    const tf::polygon<Dims, Policy0> &poly) {
  auto res = closest_metric_point_pair_impl(poly, seg);
  std::swap(res.first, res.second);
  return res;
}

template <typename Policy0, typename Policy1>
auto closest_metric_point_pair2d(const tf::polygon<2, Policy0> &poly_in,
                                 const tf::segment<2, Policy1> &seg1) {
  auto best = core::closest_metric_point_pair_impl(poly_in, seg1);
  if (best.metric == 0)
    return best;
  return min(best, closest_metric_point_pair(poly_in, seg1[1]));
}

} // namespace core

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point_pair(const tf::polygon<Dims, Policy0> &poly_in,
                               const tf::segment<Dims, Policy1> &seg1) {
  if constexpr (Dims == 2) {
    return core::closest_metric_point_pair2d(poly_in, seg1);
  } else {
    const auto &poly = tf::tag_plane(poly_in);
    auto best = core::closest_metric_point_pair_impl(poly, seg1);
    if (best.metric == 0)
      return best;
    return min(best, closest_metric_point_pair(poly, seg1[1]));
  }
}

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <typename Policy, std::size_t Dims, typename Policy0>
auto closest_metric_point_pair(const tf::segment<Dims, Policy> &seg,
                               const tf::polygon<Dims, Policy0> &poly) {
  auto res = closest_metric_point_pair(poly, seg);
  std::swap(res.first, res.second);
  return res;
}

namespace core {
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point_pair_impl(const tf::polygon<Dims, Policy0> &poly0,
                                    const tf::polygon<Dims, Policy1> &poly1) {
  std::size_t size = poly1.size();
  std::size_t prev = size - 1;
  auto best = core::closest_metric_point_pair_impl(
      poly0, tf::make_segment_between_points(poly1[prev], poly1[0]));

  for (std::size_t i = 1; i < size; prev = i++) {
    best =
        min(best,
            core::closest_metric_point_pair_impl(
                poly0, tf::make_segment_between_points(poly1[prev], poly1[i])));
    if (best.metric < tf::epsilon2<decltype(best.metric)>)
      return best;
  }

  size = poly0.size();
  prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++) {
    best = min(best, core::closest_metric_point_pair_impl(
                         tf::make_segment_between_points(poly0[prev], poly0[i]),
                         poly1));
    if (best.metric < tf::epsilon2<decltype(best.metric)>)
      return best;
  }

  return best;
}
} // namespace core

/// @ingroup core_queries
/// @brief Computes the closest @ref tf::metric_point_pair between the objects.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto closest_metric_point_pair(const tf::polygon<Dims, Policy0> &poly_in0,
                               const tf::polygon<Dims, Policy1> &poly_in1) {
  if constexpr (Dims == 2) {
    return core::closest_metric_point_pair_impl(poly_in0, poly_in1);
  } else {
    if constexpr (Dims == 3 && tf::static_size_v<Policy0> == 3 &&
                  tf::static_size_v<Policy1> == 3) {
      return tf::core::closest_points_on_triangles(poly_in0, poly_in1);
    } else if constexpr (Dims == 3) {
      if (poly_in0.size() == 3 && poly_in1.size() == 3)
        return tf::core::closest_points_on_triangles(poly_in0, poly_in1);
      else {
        const auto &poly0 = tf::tag_plane(poly_in0);
        const auto &poly1 = tf::tag_plane(poly_in1);
        return core::closest_metric_point_pair_impl(poly0, poly1);
      }
    } else {
      const auto &poly0 = tf::tag_plane(poly_in0);
      const auto &poly1 = tf::tag_plane(poly_in1);
      return core::closest_metric_point_pair_impl(poly0, poly1);
    }
  }
}
} // namespace tf

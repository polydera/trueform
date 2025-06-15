/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./dot.hpp"
#include "./implementation/line_line_check.hpp"
#include "./line.hpp"
#include "./plane.hpp"
#include "./point_like.hpp"
#include "./ray.hpp"
#include "./segment.hpp"
#include <algorithm>

namespace tf {

/// @ingroup geometry
/// @brief Computes the parametric location on a ray closest to a plane.
template <typename RealT, std::size_t Dims>
auto closest_point_parametric(const tf::ray<RealT, Dims> &ray,
                              const tf::plane<RealT, Dims> &plane) {
  auto Vd = tf::dot(plane.normal, ray.direction);
  auto V0 = tf::dot(plane.normal, ray.origin) + plane.d;
  decltype(Vd) t;
  if (std::abs(Vd) < std::numeric_limits<RealT>::epsilon()) {
    t = 0;
  } else {
    t = -V0 / Vd;
    t *= (t > 0);
  }
  return t;
}

/// @ingroup geometry
/// @brief Computes the parametric location on a line closest to a plane.
template <typename RealT, std::size_t Dims>
auto closest_point_parametric(const tf::line<RealT, Dims> &line,
                              const tf::plane<RealT, Dims> &plane) {
  auto Vd = tf::dot(plane.normal, line.direction);
  auto V0 = tf::dot(plane.normal, line.origin) + plane.d;
  decltype(Vd) t;
  if (std::abs(Vd) < std::numeric_limits<RealT>::epsilon()) {
    t = 0;
  } else {
    t = -V0 / Vd;
  }
  return t;
}

/// @ingroup geometry
/// @brief Computes the parametric location on a ray closest to a plane.
template <typename Policy, typename RealType, std::size_t Dims>
auto closest_point_parametric(const tf::segment<Policy> &segment,
                              const tf::plane<RealType, Dims> &plane) {
  auto line = tf::make_line_between_points(segment[0], segment[1]);
  auto t = closest_point_parametric(line, plane);
  return std::clamp(t, decltype(t)(0), decltype(t)(1));
}

/// @ingroup geometry
/// @brief Computes the parametric location on a segment closest to a point.
template <typename RealType, std::size_t Dims, typename T>
auto closest_point_parametric(const tf::ray<RealType, Dims> &ray,
                              const tf::point_like<Dims, T> &point) {
  auto dist_vec = point - ray.origin;
  auto t = tf::dot(dist_vec, ray.direction) / ray.direction.length2();
  return std::max(decltype(t)(0), t);
}

/// @ingroup geometry
/// @brief Computes the parametric location on a line closest to a point.
template <typename RealType, std::size_t Dims, typename T>
auto closest_point_parametric(const tf::line<RealType, Dims> &line,
                              const tf::point_like<Dims, T> &point) {
  auto dist_vec = point - line.origin;
  auto t = tf::dot(dist_vec, line.direction) / line.direction.length2();
  return t;
}

/// @ingroup geometry
/// @brief Computes the parametric location on a segment closest to a point.
template <typename Policy0, std::size_t Dims, typename T>
auto closest_point_parametric(const tf::segment<Policy0> &segment,
                              const tf::point_like<Dims, T> &point) {
  auto direction = segment[1] - segment[0];
  auto dist_vec = point - segment[0];
  auto t = tf::dot(dist_vec, direction) / direction.length2();
  return std::clamp(t, decltype(t)(0), decltype(t)(1));
}

/// @ingroup geometry
/// @brief Computes the parametric locations of closest points on a ray, and a
/// ray.
template <typename RealT, std::size_t Dims>
auto closest_point_parametric(const tf::ray<RealT, Dims> &ray0,
                              const tf::ray<RealT, Dims> &ray1) {
  auto [non_parallel, t0, t1] = tf::implementation::line_line_check(ray0, ray1);

  if (non_parallel) {
    // Clamp t0 and t1 to be non-negative for rays
    if (t0 < 0) {
      t0 = 0;
      // Recompute t1 based on clamped t0
      auto dist_vec = ray0.origin + t0 * ray0.direction - ray1.origin;
      auto dir_dot = tf::dot(ray1.direction, ray1.direction);
      t1 = tf::dot(dist_vec, ray1.direction) / dir_dot;
      t1 = std::max(t1, RealT{0});
    } else if (t1 < 0) {
      t1 = 0;
      // Recompute t0 based on clamped t1
      auto dist_vec = ray1.origin + t1 * ray1.direction - ray0.origin;
      auto dir_dot = tf::dot(ray0.direction, ray0.direction);
      t0 = tf::dot(dist_vec, ray0.direction) / dir_dot;
      t0 = std::max(t0, RealT{0});
    }
  } else {
    // Lines are parallel; set parameters based on projections
    t0 = 0;
    auto dist_vec = ray0.origin - ray1.origin;
    auto dir_dot = tf::dot(ray1.direction, ray1.direction);
    t1 = tf::dot(dist_vec, ray1.direction) / dir_dot;
    t1 = std::max(t1, RealT(0));
  }
  return std::make_pair(t0, t1);
}

/// @ingroup geometry
/// @brief Computes the parametric locations of closest points on a ray, and a
/// line.
template <typename RealT, std::size_t Dims>
auto closest_point_parametric(const tf::ray<RealT, Dims> &ray,
                              const tf::line<RealT, Dims> &line) {
  auto [non_parallel, t0, t1] = tf::implementation::line_line_check(ray, line);

  if (non_parallel) {
    // Clamp t0 and t1 to be non-negative for rays
    if (t0 < 0) {
      t0 = 0;
      // Recompute t1 based on clamped t0
      auto dist_vec = ray.origin + t0 * ray.direction - line.origin;
      auto dir_dot = tf::dot(line.direction, line.direction);
      t1 = tf::dot(dist_vec, line.direction) / dir_dot;
      t1 = std::max(t1, RealT{0});
    }
  } else {
    // Lines are parallel; set parameters based on projections
    t0 = 0;
    auto dist_vec = ray.origin - line.origin;
    auto dir_dot = tf::dot(line.direction, line.direction);
    t1 = tf::dot(dist_vec, line.direction) / dir_dot;
    t1 = std::max(t1, RealT(0));
  }
  return std::make_pair(t0, t1);
}

/// @ingroup geometry
/// @brief Computes the parametric locations of closest points on a line, and a
/// ray.
template <typename RealT, std::size_t Dims>
auto closest_point_parametric(const tf::line<RealT, Dims> &line,
                              const tf::ray<RealT, Dims> &ray) {
  auto out = closest_point_parametric(ray, line);
  std::swap(out.first, out.second);
  return out;
}

/// @ingroup geometry
/// @brief Computes the parametric locations of closest points on a ray, and a
/// segment.
template <typename RealT, std::size_t Dims, typename T>
auto closest_point_parametric(const tf::ray<RealT, Dims> &ray0,
                              const tf::segment<T> &segment) {

  auto ray1 = tf::make_line_between_points(segment[0], segment[1]);
  auto [non_parallel, t0, t1] = tf::implementation::line_line_check(ray0, ray1);

  using real_t = decltype(t0);

  if (non_parallel) {
    // Clamp t0 and t1 to be non-negative for rays
    if (t0 < 0) {
      t0 = 0;
      // Recompute t1 based on clamped t0
      auto dist_vec = ray0.origin + t0 * ray0.direction - ray1.origin;
      auto dir_dot = tf::dot(ray1.direction, ray1.direction);
      t1 = tf::dot(dist_vec, ray1.direction) / dir_dot;
    }
    t1 = std::clamp(t1, real_t{0}, real_t{1});
  } else {
    // Lines are parallel; set parameters based on projections
    t0 = 0;
    auto dist_vec = ray0.origin - ray1.origin;
    auto dir_dot = tf::dot(ray1.direction, ray1.direction);
    t1 = tf::dot(dist_vec, ray1.direction) / dir_dot;
    t1 = std::clamp(t1, real_t{0}, real_t{1});
  }
  return std::make_pair(t0, t1);
}

/// @ingroup geometry
/// @brief Computes the parametric locations of closest points on a segment, and
/// a ray.
template <typename Policy0, typename RealT, std::size_t Dims>
auto closest_point_parametric(const tf::segment<Policy0> &segment0,
                              const tf::ray<RealT, Dims> &ray1) {
  auto out = closest_point_parametric(ray1, segment0);
  std::swap(out.first, out.second);
  return out;
}

/// @ingroup geometry
/// @brief Computes the parametric locations of closest points on a line, and a
/// segment.
template <typename RealT, std::size_t Dims, typename Policy0>
auto closest_point_parametric(const tf::line<RealT, Dims> &line,
                              const tf::segment<Policy0> &segment) {
  auto line_segment = tf::make_line_between_points(segment[0], segment[1]);
  auto [non_parallel, t0, t1] =
      tf::implementation::line_line_check(line, line_segment);
  using real_t = decltype(t0);

  if (non_parallel) {
    // Clamp t1 to [0, 1] for segment
    t1 = std::clamp(t1, real_t(0), real_t(1));
    // Recompute t0 based on clamped t1
    auto dist_vec =
        line_segment.origin + t1 * line_segment.direction - line.origin;
    real_t dir_dot = tf::dot(line.direction, line.direction);
    t0 = tf::dot(dist_vec, line.direction) / dir_dot;
  } else {
    // Lines are parallel; set parameters based on projections
    auto dist_vec = segment[0] - line.origin;
    real_t dir_dot = tf::dot(line.direction, line.direction);
    t0 = tf::dot(dist_vec, line.direction) / dir_dot;
    t1 = real_t(0);
  }

  return std::make_pair(t0, t1);
}

/// @ingroup geometry
/// @brief Computes the parametric locations of closest points on a segment, and
/// a line.
template <typename Policy0, typename RealT, std::size_t Dims>
auto closest_point_parametric(const tf::segment<Policy0> &segment,
                              const tf::line<RealT, Dims> &line) {
  auto result = closest_point_parametric(line, segment);
  return std::make_pair(result.second, result.first);
}

/// @ingroup geometry
/// @brief Computes the parametric locations of closest points on a line, and a
/// line.
template <typename RealT, std::size_t Dims>
auto closest_point_parametric(const tf::line<RealT, Dims> &line0,
                              const tf::line<RealT, Dims> &line1) {

  auto [non_parallel, t0, t1] =
      tf::implementation::line_line_check(line0, line1);
  using real_t = decltype(t0);
  if (!non_parallel) {
    // Lines are parallel — choose t0 = 0, project line0.origin onto line1
    t0 = real_t{0};

    auto diff = line0.origin - line1.origin;
    auto dir_dot = tf::dot(line1.direction, line1.direction);

    if (dir_dot > real_t{0}) {
      t1 = tf::dot(diff, line1.direction) / dir_dot;
    } else {
      t1 = real_t{0}; // degenerate case: direction is zero vector
    }
  }
  return std::make_pair(t0, t1);
}

/// @ingroup geometry
/// @brief Computes the parametric locations of closest points on a segment, and
/// a segment.
template <typename Policy0, typename Policy1>
auto closest_point_parametric(const tf::segment<Policy0> &segment0,
                              const tf::segment<Policy1> &segment1) {
  using real_t = std::decay_t<decltype(segment0[0][0])>;
  auto line0 = tf::make_line_between_points(segment0[0], segment0[1]);
  auto line1 = tf::make_line_between_points(segment1[0], segment1[1]);

  auto [non_parallel, t0, t1] =
      tf::implementation::line_line_check(line0, line1);

  if (non_parallel) {
    // Clamp t0 and t1 to [0, 1] for segments
    t0 = std::clamp(t0, real_t(0), real_t(1));
    t1 = std::clamp(t1, real_t(0), real_t(1));
  } else {
    // Lines are parallel; find the closest endpoints
    real_t min_dist_sq = (segment0[0] - segment1[0]).length2();
    std::pair<real_t, real_t> min_pair{0, 0};

    auto check_f = [&](const auto &p0, const auto &p1, auto candidate) {
      auto dist_sq = (p0 - p1).length2();
      if (dist_sq < min_dist_sq) {
        min_dist_sq = dist_sq;
        min_pair = candidate;
      }
    };
    check_f(segment0[0], segment1[1], std::make_pair(real_t(0), real_t(1)));
    check_f(segment0[1], segment1[0], std::make_pair(real_t(1), real_t(0)));
    check_f(segment0[1], segment1[1], std::make_pair(real_t(1), real_t(1)));
    t0 = min_pair.first;
    t1 = min_pair.second;
  }

  return std::make_pair(t0, t1);
}
} // namespace tf

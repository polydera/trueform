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

#include "../../core/aabb_like.hpp"
#include "../../core/distance.hpp"
#include "../../core/line_like.hpp"
#include "../../core/obb_like.hpp"
#include "../../core/obbrss_like.hpp"
#include "../../core/plane_like.hpp"
#include "../../core/point_like.hpp"
#include "../../core/ray_like.hpp"
#include "../../core/rss_from.hpp"
#include "../../core/rss_like.hpp"
#include "../../core/segment.hpp"

namespace tf::spatial {

/// @brief Compute traversal metric for single-tree queries.
///
/// Returns squared minimum distance lower bound for pruning.
///
/// @return Squared distance lower bound
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::aabb_like<Dims, Policy0> &aabb,
                      const tf::aabb_like<Dims, Policy1> &other) {
  return tf::distance2(aabb, other);
}

/// @brief Compute traversal metric for single-tree queries.
///
/// @return Squared distance lower bound
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::aabb_like<Dims, Policy0> &aabb,
                      const tf::point_like<Dims, Policy1> &pt) {
  return tf::distance2(aabb, pt);
}

/// @brief Compute traversal metric for single-tree queries.
///
/// Uses bounding sphere for lower bound on distance to line.
///
/// @return Squared distance lower bound
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::aabb_like<Dims, Policy0> &aabb,
                      const tf::line_like<Dims, Policy1> &line) {
  using T = tf::coordinate_type<Policy0, Policy1>;

  auto center = aabb.center();
  auto half_extent = (aabb.max - aabb.min) * T(0.5);
  auto r = half_extent.length();

  auto d_center = tf::distance(line, center);
  auto result = std::max(T(0), d_center - r);
  return result * result;
}

/// @brief Compute traversal metric for single-tree queries.
///
/// Uses bounding sphere for lower bound on distance to ray.
///
/// @return Squared distance lower bound
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::aabb_like<Dims, Policy0> &aabb,
                      const tf::ray_like<Dims, Policy1> &ray) {
  using T = tf::coordinate_type<Policy0, Policy1>;

  auto center = aabb.center();
  auto half_extent = (aabb.max - aabb.min) * T(0.5);
  auto r = half_extent.length();

  auto d_center = tf::distance(ray, center);
  auto result = std::max(T(0), d_center - r);
  return result * result;
}

/// @brief Compute traversal metric for single-tree queries.
///
/// @return Squared distance (exact)
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::aabb_like<Dims, Policy0> &aabb,
                      const tf::plane_like<Dims, Policy1> &plane) {
  return tf::distance2(aabb, plane);
}

// ============================================================================
// OBB row
// ============================================================================

/// @brief Compute traversal metric for single-tree queries.
///
/// Converts both OBB and AABB to RSS for distance bound.
///
/// @return Squared distance lower bound
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::obb_like<Dims, Policy0> &obb,
                      const tf::aabb_like<Dims, Policy1> &aabb) {
  static_assert(Dims == 3,
                "traversal_metric(obb, aabb) is implemented for 3D only.");

  auto obb_rss = tf::rss_from(obb);
  auto aabb_rss = tf::rss_from(aabb);

  return tf::distance2(obb_rss, aabb_rss);
}

/// @brief Compute traversal metric for single-tree queries.
///
/// @return Squared distance (exact)
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::obb_like<Dims, Policy0> &obb,
                      const tf::point_like<Dims, Policy1> &pt) {
  return tf::distance2(obb, pt);
}

/// @brief Compute traversal metric for single-tree queries.
///
/// Uses bounding capsule along longest axis for lower bound on distance to line.
///
/// @return Squared distance lower bound
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::obb_like<Dims, Policy0> &obb,
                      const tf::line_like<Dims, Policy1> &line) {
  static_assert(Dims == 2 || Dims == 3,
                "traversal_metric(obb, line) is implemented for 2D and 3D only.");
  using T = tf::coordinate_type<Policy0, Policy1>;

  // OBB center
  auto center = obb.origin;
  for (std::size_t i = 0; i < Dims; ++i) {
    center = center + obb.axes[i] * (obb.extent[i] * T(0.5));
  }

  // Segment along longest axis (axes[0]) through center
  auto half_len = obb.extent[0] * T(0.5);
  auto p0 = center - obb.axes[0] * half_len;
  auto p1 = center + obb.axes[0] * half_len;
  auto seg = tf::make_segment_between_points(p0, p1);

  // Capsule radius: half-diagonal of cross-section
  T r;
  if constexpr (Dims == 2) {
    r = obb.extent[1] * T(0.5);
  } else {
    r = tf::sqrt(obb.extent[1] * obb.extent[1] +
                 obb.extent[2] * obb.extent[2]) *
        T(0.5);
  }

  auto d_seg = tf::distance(seg, line);
  auto result = std::max(T(0), d_seg - r);
  return result * result;
}

/// @brief Compute traversal metric for single-tree queries.
///
/// Uses bounding capsule along longest axis for lower bound on distance to ray.
///
/// @return Squared distance lower bound
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::obb_like<Dims, Policy0> &obb,
                      const tf::ray_like<Dims, Policy1> &ray) {
  static_assert(Dims == 2 || Dims == 3,
                "traversal_metric(obb, ray) is implemented for 2D and 3D only.");
  using T = tf::coordinate_type<Policy0, Policy1>;

  // OBB center
  auto center = obb.origin;
  for (std::size_t i = 0; i < Dims; ++i) {
    center = center + obb.axes[i] * (obb.extent[i] * T(0.5));
  }

  // Segment along longest axis (axes[0]) through center
  auto half_len = obb.extent[0] * T(0.5);
  auto p0 = center - obb.axes[0] * half_len;
  auto p1 = center + obb.axes[0] * half_len;
  auto seg = tf::make_segment_between_points(p0, p1);

  // Capsule radius: half-diagonal of cross-section
  T r;
  if constexpr (Dims == 2) {
    r = obb.extent[1] * T(0.5);
  } else {
    r = tf::sqrt(obb.extent[1] * obb.extent[1] +
                 obb.extent[2] * obb.extent[2]) *
        T(0.5);
  }

  auto d_seg = tf::distance(seg, ray);
  auto result = std::max(T(0), d_seg - r);
  return result * result;
}

/// @brief Compute traversal metric for single-tree queries.
///
/// @return Squared distance (exact)
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::obb_like<Dims, Policy0> &obb,
                      const tf::plane_like<Dims, Policy1> &plane) {
  return tf::distance2(obb, plane);
}

// ============================================================================
// RSS row
// ============================================================================

/// @brief Compute traversal metric for single-tree queries.
///
/// Converts AABB to RSS for distance bound.
///
/// @return Squared distance lower bound
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::rss_like<Dims, Policy0> &rss,
                      const tf::aabb_like<Dims, Policy1> &aabb) {
  static_assert(Dims == 3,
                "traversal_metric(rss, aabb) is implemented for 3D only.");

  auto aabb_rss = tf::rss_from(aabb);
  return tf::distance2(rss, aabb_rss);
}

/// @brief Compute traversal metric for single-tree queries.
///
/// @return Squared distance (exact)
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::rss_like<Dims, Policy0> &rss,
                      const tf::point_like<Dims, Policy1> &pt) {
  return tf::distance2(rss, pt);
}

/// @brief Compute traversal metric for single-tree queries.
///
/// Uses bounding capsule along principal axis for lower bound on distance to line.
///
/// @return Squared distance lower bound
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::rss_like<Dims, Policy0> &rss,
                      const tf::line_like<Dims, Policy1> &line) {
  static_assert(Dims == 2 || Dims == 3,
                "traversal_metric(rss, line) is implemented for 2D and 3D only.");
  using T = tf::coordinate_type<Policy0, Policy1>;

  auto center = rss.center();
  auto half_len = rss.length[0] * T(0.5);

  // Capsule radius: perpendicular half-extent + sweep radius
  // 2D: no perpendicular length, just radius
  // 3D: length[1]/2 + radius (axes ordered by size, so length[1] <= length[0])
  T r;
  if constexpr (Dims == 2) {
    r = rss.radius;
  } else {
    r = rss.length[1] * T(0.5) + rss.radius;
  }

  auto p0 = center - rss.axes[0] * half_len;
  auto p1 = center + rss.axes[0] * half_len;
  auto seg = tf::make_segment_between_points(p0, p1);

  auto d_seg = tf::distance(seg, line);
  auto result = std::max(T(0), d_seg - r);
  return result * result;
}

/// @brief Compute traversal metric for single-tree queries.
///
/// Uses bounding capsule along principal axis for lower bound on distance to ray.
///
/// @return Squared distance lower bound
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::rss_like<Dims, Policy0> &rss,
                      const tf::ray_like<Dims, Policy1> &ray) {
  static_assert(Dims == 2 || Dims == 3,
                "traversal_metric(rss, ray) is implemented for 2D and 3D only.");
  using T = tf::coordinate_type<Policy0, Policy1>;

  auto center = rss.center();
  auto half_len = rss.length[0] * T(0.5);

  // Capsule radius: perpendicular half-extent + sweep radius
  T r;
  if constexpr (Dims == 2) {
    r = rss.radius;
  } else {
    r = rss.length[1] * T(0.5) + rss.radius;
  }

  auto p0 = center - rss.axes[0] * half_len;
  auto p1 = center + rss.axes[0] * half_len;
  auto seg = tf::make_segment_between_points(p0, p1);

  auto d_seg = tf::distance(seg, ray);
  auto result = std::max(T(0), d_seg - r);
  return result * result;
}

/// @brief Compute traversal metric for single-tree queries.
///
/// @return Squared distance (exact)
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::rss_like<Dims, Policy0> &rss,
                      const tf::plane_like<Dims, Policy1> &plane) {
  return tf::distance2(rss, plane);
}

// ============================================================================
// OBBRSS row (forwards to RSS)
// ============================================================================

/// @brief Compute traversal metric for single-tree queries.
///
/// Forwards to RSS metric.
///
/// @return Squared distance lower bound
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::obbrss_like<Dims, Policy0> &obbrss,
                      const tf::aabb_like<Dims, Policy1> &aabb) {
  auto rss = tf::make_rss_like(obbrss.rss_origin, obbrss.axes, obbrss.length,
                               obbrss.radius);
  return traversal_metric(rss, aabb);
}

/// @brief Compute traversal metric for single-tree queries.
///
/// Forwards to OBB metric (cheaper than RSS, no sqrt needed).
///
/// @return Squared distance lower bound
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::obbrss_like<Dims, Policy0> &obbrss,
                      const tf::point_like<Dims, Policy1> &pt) {
  auto obb = tf::make_obb_like(obbrss.obb_origin, obbrss.axes, obbrss.extent);
  return traversal_metric(obb, pt);
}

/// @brief Compute traversal metric for single-tree queries.
///
/// Forwards to RSS metric.
///
/// @return Squared distance lower bound
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::obbrss_like<Dims, Policy0> &obbrss,
                      const tf::line_like<Dims, Policy1> &line) {
  auto rss = tf::make_rss_like(obbrss.rss_origin, obbrss.axes, obbrss.length,
                               obbrss.radius);
  return traversal_metric(rss, line);
}

/// @brief Compute traversal metric for single-tree queries.
///
/// Forwards to RSS metric.
///
/// @return Squared distance lower bound
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::obbrss_like<Dims, Policy0> &obbrss,
                      const tf::ray_like<Dims, Policy1> &ray) {
  auto rss = tf::make_rss_like(obbrss.rss_origin, obbrss.axes, obbrss.length,
                               obbrss.radius);
  return traversal_metric(rss, ray);
}

/// @brief Compute traversal metric for single-tree queries.
///
/// Forwards to RSS metric.
///
/// @return Squared distance (exact)
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metric(const tf::obbrss_like<Dims, Policy0> &obbrss,
                      const tf::plane_like<Dims, Policy1> &plane) {
  auto rss = tf::make_rss_like(obbrss.rss_origin, obbrss.axes, obbrss.length,
                               obbrss.radius);
  return traversal_metric(rss, plane);
}

} // namespace tf::spatial

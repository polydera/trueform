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
#include "../../core/obb_like.hpp"
#include "../../core/obbrss_like.hpp"
#include "../../core/rss_from.hpp"
#include "../../core/rss_like.hpp"
#include <utility>

namespace tf::spatial {

// ============================================================================
// AABB row
// ============================================================================

/// @brief Compute traversal metrics for dual-tree queries.
///
/// Returns {min_d2, max_d2} where:
/// - min_d2: squared minimum distance between bounding volumes
/// - max_d2: upper bound on squared distance for pruning
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::aabb_like<Dims, Policy0> &a,
                       const tf::aabb_like<Dims, Policy1> &b) {
  auto min_d2 = tf::distance2(a, b);
  auto max_d2 = std::max((a.min - b.min).length2(), (a.max - b.max).length2());
  return std::make_pair(min_d2, max_d2);
}

// ============================================================================
// RSS row (core implementation - all others forward here)
// ============================================================================

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::rss_like<Dims, Policy0> &rss0,
                       const tf::rss_like<Dims, Policy1> &rss1) {
  static_assert(Dims == 2 || Dims == 3,
                "traversal_metrics(rss) is implemented for 2D and 3D only.");

  auto min_d2 = tf::distance2(rss0, rss1);

  // Use the dimension-generic center() method
  auto center0 = rss0.center();
  auto center1 = rss1.center();
  auto max_d2 = std::max(min_d2, (center1 - center0).length2());

  return std::make_pair(min_d2, max_d2);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::rss_like<Dims, Policy0> &rss,
                       const tf::aabb_like<Dims, Policy1> &aabb) {
  static_assert(Dims == 3,
                "traversal_metrics(rss, aabb) is implemented for 3D only.");

  auto aabb_rss = tf::rss_from(aabb);
  return traversal_metrics(rss, aabb_rss);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::aabb_like<Dims, Policy0> &aabb,
                       const tf::rss_like<Dims, Policy1> &rss) {
  return traversal_metrics(rss, aabb);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::rss_like<Dims, Policy0> &rss,
                       const tf::obb_like<Dims, Policy1> &obb) {
  static_assert(Dims == 3,
                "traversal_metrics(rss, obb) is implemented for 3D only.");

  auto obb_rss = tf::rss_from(obb);
  return traversal_metrics(rss, obb_rss);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::obb_like<Dims, Policy0> &obb,
                       const tf::rss_like<Dims, Policy1> &rss) {
  return traversal_metrics(rss, obb);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::rss_like<Dims, Policy0> &rss,
                       const tf::obbrss_like<Dims, Policy1> &obbrss) {
  auto obbrss_rss = tf::make_rss_like(obbrss.rss_origin, obbrss.axes,
                                      obbrss.length, obbrss.radius);
  return traversal_metrics(rss, obbrss_rss);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::obbrss_like<Dims, Policy0> &obbrss,
                       const tf::rss_like<Dims, Policy1> &rss) {
  return traversal_metrics(rss, obbrss);
}

// ============================================================================
// OBB row (forwards to RSS)
// ============================================================================

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::obb_like<Dims, Policy0> &obb0,
                       const tf::obb_like<Dims, Policy1> &obb1) {
  auto rss0 = tf::rss_from(obb0);
  auto rss1 = tf::rss_from(obb1);
  return traversal_metrics(rss0, rss1);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::obb_like<Dims, Policy0> &obb,
                       const tf::aabb_like<Dims, Policy1> &aabb) {
  auto obb_rss = tf::rss_from(obb);
  auto aabb_rss = tf::rss_from(aabb);
  return traversal_metrics(obb_rss, aabb_rss);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::aabb_like<Dims, Policy0> &aabb,
                       const tf::obb_like<Dims, Policy1> &obb) {
  return traversal_metrics(obb, aabb);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::obb_like<Dims, Policy0> &obb,
                       const tf::obbrss_like<Dims, Policy1> &obbrss) {
  auto obb_rss = tf::rss_from(obb);
  auto obbrss_rss = tf::make_rss_like(obbrss.rss_origin, obbrss.axes,
                                      obbrss.length, obbrss.radius);
  return traversal_metrics(obb_rss, obbrss_rss);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::obbrss_like<Dims, Policy0> &obbrss,
                       const tf::obb_like<Dims, Policy1> &obb) {
  return traversal_metrics(obb, obbrss);
}

// ============================================================================
// OBBRSS row (forwards to RSS)
// ============================================================================

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::obbrss_like<Dims, Policy0> &a,
                       const tf::obbrss_like<Dims, Policy1> &b) {
  auto rss_a = tf::make_rss_like(a.rss_origin, a.axes, a.length, a.radius);
  auto rss_b = tf::make_rss_like(b.rss_origin, b.axes, b.length, b.radius);
  return traversal_metrics(rss_a, rss_b);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::obbrss_like<Dims, Policy0> &obbrss,
                       const tf::aabb_like<Dims, Policy1> &aabb) {
  auto obbrss_rss = tf::make_rss_like(obbrss.rss_origin, obbrss.axes,
                                      obbrss.length, obbrss.radius);
  auto aabb_rss = tf::rss_from(aabb);
  return traversal_metrics(obbrss_rss, aabb_rss);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::aabb_like<Dims, Policy0> &aabb,
                       const tf::obbrss_like<Dims, Policy1> &obbrss) {
  return traversal_metrics(obbrss, aabb);
}

} // namespace tf::spatial

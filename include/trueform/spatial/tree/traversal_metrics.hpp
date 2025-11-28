/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
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

/// @brief Compute traversal metrics for dual-tree queries.
///
/// Returns {min_d2, max_d2} where:
/// - min_d2: squared minimum distance between bounding volumes
/// - max_d2: upper bound on squared distance for pruning
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::obbrss_like<Dims, Policy0> &a,
                       const tf::obbrss_like<Dims, Policy1> &b) {
  static_assert(Dims == 3, "traversal_metrics(obbrss) is implemented for 3D only.");
  using T = tf::coordinate_type<Policy0, Policy1>;

  auto min_d2 = tf::distance2(a, b);

  auto center0 = a.rss_origin + a.axes[0] * (a.length[0] * T(0.5)) +
                 a.axes[1] * (a.length[1] * T(0.5));
  auto center1 = b.rss_origin + b.axes[0] * (b.length[0] * T(0.5)) +
                 b.axes[1] * (b.length[1] * T(0.5));
  auto max_d2 = std::max(min_d2, (center1 - center0).length2());

  return std::make_pair(min_d2, max_d2);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// Converts OBB to RSS (rectangle at midplane, radius = extent[2]/2)
/// for tight distance bounds.
///
/// Returns {min_d2, max_d2} where:
/// - min_d2: squared minimum distance between bounding volumes
/// - max_d2: upper bound on squared distance for pruning
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::obb_like<Dims, Policy0> &obb0,
                       const tf::obb_like<Dims, Policy1> &obb1) {
  static_assert(Dims == 3, "traversal_metrics(obb) is implemented for 3D only.");
  using T = tf::coordinate_type<Policy0, Policy1>;

  auto rss0 = tf::rss_from(obb0);
  auto rss1 = tf::rss_from(obb1);

  auto min_d2 = tf::distance2(rss0, rss1);

  auto center0 = rss0.origin + obb0.axes[0] * (obb0.extent[0] * T(0.5)) +
                 obb0.axes[1] * (obb0.extent[1] * T(0.5));
  auto center1 = rss1.origin + obb1.axes[0] * (obb1.extent[0] * T(0.5)) +
                 obb1.axes[1] * (obb1.extent[1] * T(0.5));
  auto max_d2 = std::max(min_d2, (center1 - center0).length2());

  return std::make_pair(min_d2, max_d2);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// Converts OBB to RSS for tight distance bounds against OBBRSS.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::obb_like<Dims, Policy0> &obb,
                       const tf::obbrss_like<Dims, Policy1> &obbrss) {
  static_assert(Dims == 3, "traversal_metrics(obb, obbrss) is implemented for 3D only.");
  using T = tf::coordinate_type<Policy0, Policy1>;

  auto obb_rss = tf::rss_from(obb);

  // Use exact RSS from OBBRSS
  auto obbrss_rss = tf::make_rss_like(obbrss.rss_origin, obbrss.axes,
                                      obbrss.length, obbrss.radius);

  auto min_d2 = tf::distance2(obb_rss, obbrss_rss);

  auto center0 = obb_rss.origin + obb.axes[0] * (obb.extent[0] * T(0.5)) +
                 obb.axes[1] * (obb.extent[1] * T(0.5));
  auto center1 = obbrss.rss_origin + obbrss.axes[0] * (obbrss.length[0] * T(0.5)) +
                 obbrss.axes[1] * (obbrss.length[1] * T(0.5));
  auto max_d2 = std::max(min_d2, (center1 - center0).length2());

  return std::make_pair(min_d2, max_d2);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::obbrss_like<Dims, Policy0> &obbrss,
                       const tf::obb_like<Dims, Policy1> &obb) {
  return traversal_metrics(obb, obbrss);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// Converts both AABB and OBB to RSS for distance bounds.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::aabb_like<Dims, Policy0> &aabb,
                       const tf::obb_like<Dims, Policy1> &obb) {
  static_assert(Dims == 3, "traversal_metrics(aabb, obb) is implemented for 3D only.");
  using T = tf::coordinate_type<Policy0, Policy1>;

  auto aabb_rss = tf::rss_from(aabb);
  auto obb_rss = tf::rss_from(obb);

  auto min_d2 = tf::distance2(aabb_rss, obb_rss);

  auto center0 = aabb_rss.origin + aabb_rss.axes[0] * (aabb_rss.length[0] * T(0.5)) +
                 aabb_rss.axes[1] * (aabb_rss.length[1] * T(0.5));
  auto center1 = obb_rss.origin + obb.axes[0] * (obb.extent[0] * T(0.5)) +
                 obb.axes[1] * (obb.extent[1] * T(0.5));
  auto max_d2 = std::max(min_d2, (center1 - center0).length2());

  return std::make_pair(min_d2, max_d2);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::obb_like<Dims, Policy0> &obb,
                       const tf::aabb_like<Dims, Policy1> &aabb) {
  return traversal_metrics(aabb, obb);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// Converts AABB to RSS for distance bounds against OBBRSS.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::aabb_like<Dims, Policy0> &aabb,
                       const tf::obbrss_like<Dims, Policy1> &obbrss) {
  static_assert(Dims == 3, "traversal_metrics(aabb, obbrss) is implemented for 3D only.");
  using T = tf::coordinate_type<Policy0, Policy1>;

  auto aabb_rss = tf::rss_from(aabb);

  // Use exact RSS from OBBRSS
  auto obbrss_rss = tf::make_rss_like(obbrss.rss_origin, obbrss.axes,
                                      obbrss.length, obbrss.radius);

  auto min_d2 = tf::distance2(aabb_rss, obbrss_rss);

  auto center0 = aabb_rss.origin + aabb_rss.axes[0] * (aabb_rss.length[0] * T(0.5)) +
                 aabb_rss.axes[1] * (aabb_rss.length[1] * T(0.5));
  auto center1 = obbrss.rss_origin + obbrss.axes[0] * (obbrss.length[0] * T(0.5)) +
                 obbrss.axes[1] * (obbrss.length[1] * T(0.5));
  auto max_d2 = std::max(min_d2, (center1 - center0).length2());

  return std::make_pair(min_d2, max_d2);
}

/// @brief Compute traversal metrics for dual-tree queries.
///
/// @return std::pair<T, T> with {min_d2, max_d2}
template <std::size_t Dims, typename Policy0, typename Policy1>
auto traversal_metrics(const tf::obbrss_like<Dims, Policy0> &obbrss,
                       const tf::aabb_like<Dims, Policy1> &aabb) {
  return traversal_metrics(aabb, obbrss);
}

} // namespace tf::spatial

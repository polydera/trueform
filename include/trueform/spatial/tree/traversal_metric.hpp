/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../../core/aabb_like.hpp"
#include "../../core/distance.hpp"
#include "../../core/line_like.hpp"
#include "../../core/point_like.hpp"
#include "../../core/ray_like.hpp"

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

} // namespace tf::spatial

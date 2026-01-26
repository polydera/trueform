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
#include "../aabb.hpp"
#include "../dot.hpp"
#include "../epsilon.hpp"
#include "../intersects.hpp"
#include "../obb_like.hpp"
#include "../plane_like.hpp"
#include "../point.hpp"
#include "../polygon.hpp"
#include "../ray_cast.hpp"
#include "../ray_config.hpp"
#include "../segment.hpp"
#include <limits>

namespace tf::core {

/// @brief Check if ray intersects OBB
template <std::size_t Dims, typename Policy0, typename Policy1>
auto obb_intersects_ray(const tf::obb_like<Dims, Policy0> &obb,
                        const tf::ray_like<Dims, Policy1> &ray) -> bool {
  return static_cast<bool>(tf::ray_cast(ray, obb));
}

/// @brief Check if line intersects OBB
template <std::size_t Dims, typename Policy0, typename Policy1>
auto obb_intersects_line(const tf::obb_like<Dims, Policy0> &obb,
                         const tf::line_like<Dims, Policy1> &line) -> bool {
  using T = tf::coordinate_type<Policy0, Policy1>;
  auto ray = tf::make_ray_like(line.origin, line.direction);
  return static_cast<bool>(
      tf::ray_cast(ray, obb,
                   tf::make_ray_config(std::numeric_limits<T>::lowest(),
                                       std::numeric_limits<T>::max())));
}

/// @brief Check if segment intersects OBB
template <std::size_t Dims, typename Policy0, typename Policy1>
auto obb_intersects_segment(const tf::obb_like<Dims, Policy0> &obb,
                            const tf::segment<Dims, Policy1> &seg) -> bool {
  using T = tf::coordinate_type<Policy0, Policy1>;
  auto ray = tf::make_ray_between_points(seg[0], seg[1]);
  return static_cast<bool>(
      tf::ray_cast(ray, obb, tf::make_ray_config(T(0), T(1))));
}

/// @brief Check if plane intersects OBB
template <std::size_t Dims, typename Policy0, typename Policy1>
auto obb_intersects_plane(const tf::obb_like<Dims, Policy0> &obb,
                          const tf::plane_like<Dims, Policy1> &plane) -> bool {
  using T = tf::coordinate_type<Policy0, Policy1>;

  T base = tf::dot(obb.origin, plane.normal) + plane.d;
  T n_min = base, n_max = base;

  for (std::size_t i = 0; i < Dims; ++i) {
    T proj = obb.extent[i] * tf::dot(obb.axes[i], plane.normal);
    if (proj > T(0))
      n_max += proj;
    else
      n_min += proj;
  }

  return n_min <= tf::epsilon<T> && n_max >= -tf::epsilon<T>;
}

/// @brief Check if polygon intersects OBB
/// Uses transform-and-carry pattern to avoid allocation and redundant
/// transforms
template <std::size_t Dims, typename Policy0, typename Policy1>
auto obb_intersects_polygon(const tf::obb_like<Dims, Policy0> &obb,
                            const tf::polygon<Dims, Policy1> &poly) -> bool {
  using T = tf::coordinate_type<Policy0, Policy1>;

  std::size_t size = poly.size();
  tf::aabb<T, Dims> local_aabb;
  for (std::size_t d = 0; d < Dims; ++d) {
    local_aabb.min[d] = T(0);
    local_aabb.max[d] = obb.extent[d];
  }

  // Helper to transform point to OBB local coordinates
  auto to_local = [&](const auto &pt) {
    tf::point<T, Dims> local;
    auto diff = pt - obb.origin;
    for (std::size_t d = 0; d < Dims; ++d) {
      local[d] = tf::dot(diff, obb.axes[d]);
    }
    return local;
  };

  // Transform initial prev vertex (last vertex of polygon)
  std::size_t prev = size - 1;
  auto local_prev = to_local(poly[prev]);

  // Iterate edges, carrying forward the transformed vertex
  for (std::size_t i = 0; i < size; prev = i++) {
    auto local_curr = to_local(poly[i]);
    if (intersects(tf::make_segment_between_points(local_prev, local_curr),
                   local_aabb))
      return true;
    local_prev = local_curr;
  }
  return false;
}

} // namespace tf::core

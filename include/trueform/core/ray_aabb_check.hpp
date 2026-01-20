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
#include "./intersect_status.hpp"
#include "./ray_like.hpp"
#include "./vector_like.hpp"
#include <algorithm>

namespace tf::core {

template <std::size_t Dims, typename Policy0, typename T, typename Policy1,
          typename RealT>
auto ray_aabb_check(const tf::ray_like<Dims, Policy0> &ray,
                    const tf::vector_like<Dims, T> &ray_dir_inv,
                    const tf::aabb_like<Dims, Policy1> &bounding_box,
                    RealT &t_min, RealT &t_max, RealT min_t, RealT max_t) {
  auto &&min = bounding_box.min;
  auto &&max = bounding_box.max;
  auto safe_max = [](auto a, auto b) { return a > b ? a : b; };
  auto safe_min = [](auto a, auto b) { return a < b ? a : b; };
  for (std::size_t i = 0; i < Dims; ++i) {
    auto min_i = min[i];
    auto max_i = max[i];
    if (ray_dir_inv[i] < 0)
      std::swap(min_i, max_i);
    auto t0 = (min_i - ray.origin[i]) * ray_dir_inv[i];
    auto t1 = (max_i - ray.origin[i]) * ray_dir_inv[i];
    t1 *= RealT(1) +
          std::copysign(RealT(2) * std::numeric_limits<RealT>::epsilon(), t1);
    min_t = safe_max(t0, min_t);
    max_t = safe_min(t1, max_t);
  }
  if (min_t <= max_t) {
    t_min = min_t;
    t_max = max_t;
    return intersect_status::intersection;
  }
  return intersect_status::none;
}
} // namespace tf::core

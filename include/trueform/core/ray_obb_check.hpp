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
#include "./dot.hpp"
#include "./epsilon_inverse.hpp"
#include "./intersect_status.hpp"
#include "./obb_like.hpp"
#include "./ray_like.hpp"
#include <algorithm>

namespace tf::core {

template <std::size_t Dims, typename Policy0, typename Policy1, typename RealT>
auto ray_obb_check(const tf::ray_like<Dims, Policy0> &ray,
                   const tf::obb_like<Dims, Policy1> &obb, RealT &t_min,
                   RealT &t_max, RealT min_t, RealT max_t) {
  auto diff = ray.origin - obb.origin;
  auto safe_max = [](auto a, auto b) { return a > b ? a : b; };
  auto safe_min = [](auto a, auto b) { return a < b ? a : b; };
  for (std::size_t i = 0; i < Dims; ++i) {
    auto local_origin_i = tf::dot(diff, obb.axes[i]);
    auto local_dir_i = tf::dot(ray.direction, obb.axes[i]);
    auto local_inv_dir_i = tf::epsilon_inverse(local_dir_i);
    RealT min_i = RealT(0);
    RealT max_i = obb.extent[i];
    if (local_inv_dir_i < 0)
      std::swap(min_i, max_i);
    auto t0 = (min_i - local_origin_i) * local_inv_dir_i;
    auto t1 = (max_i - local_origin_i) * local_inv_dir_i;
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

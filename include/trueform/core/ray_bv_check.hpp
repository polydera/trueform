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
#include "./obb_like.hpp"
#include "./obbrss.hpp"
#include "./ray_aabb_check.hpp"
#include "./ray_like.hpp"
#include "./ray_obb_check.hpp"
#include "./vector_like.hpp"

namespace tf::core {

/// @brief Check if a ray intersects an AABB.
template <std::size_t Dims, typename RayPolicy, typename T, typename BVPolicy,
          typename RealT>
auto ray_bv_check(const tf::ray_like<Dims, RayPolicy> &ray,
                  const tf::vector_like<Dims, T> &ray_inv_dir,
                  const tf::aabb_like<Dims, BVPolicy> &bv, RealT &t_min,
                  RealT &t_max, RealT min_t, RealT max_t) {
  return ray_aabb_check(ray, ray_inv_dir, bv, t_min, t_max, min_t, max_t);
}

/// @brief Check if a ray intersects an OBB.
template <std::size_t Dims, typename RayPolicy, typename T, typename BVPolicy,
          typename RealT>
auto ray_bv_check(const tf::ray_like<Dims, RayPolicy> &ray,
                  const tf::vector_like<Dims, T> &, // ray_inv_dir unused
                  const tf::obb_like<Dims, BVPolicy> &bv, RealT &t_min,
                  RealT &t_max, RealT min_t, RealT max_t) {
  return ray_obb_check(ray, bv, t_min, t_max, min_t, max_t);
}

/// @brief Check if a ray intersects an OBBRSS.
template <std::size_t Dims, typename RayPolicy, typename T, typename RealT>
auto ray_bv_check(const tf::ray_like<Dims, RayPolicy> &ray,
                  const tf::vector_like<Dims, T> &, // ray_inv_dir unused
                  const tf::obbrss<RealT, Dims> &bv, RealT &t_min, RealT &t_max,
                  RealT min_t, RealT max_t) {
  auto obb = tf::make_obb_like(bv.obb_origin, bv.axes, bv.extent);
  return ray_obb_check(ray, obb, t_min, t_max, min_t, max_t);
}

} // namespace tf::core

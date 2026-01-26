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
#include "../core/ray_hit_info.hpp"
#include "./ray_cast.hpp"

namespace tf {

/// @ingroup spatial_queries
/// @brief Cast a ray and get the hit point.
///
/// Like @ref tf::ray_cast but also computes the actual hit point on the
/// primitive surface.
///
/// @param ray The ray to cast.
/// @param form The spatial form to query.
/// @param config Optional ray configuration (min/max t values).
/// @return A @ref tf::tree_ray_info containing the hit primitive ID,
///         intersection parameter t, and the hit point coordinates.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<Dims, Policy0> &ray, const tf::form<Dims, Policy1> &form,
    tf::ray_config<tf::coordinate_type<Policy0, Policy1>> config = {}) {
  auto result = ray_cast(ray, form, config);
  using tree_policy = typename Policy1::tree_policy;
  using Index = typename tree_policy::index_type;
  tf::tree_ray_info<
      Index, tf::ray_hit_info<tf::coordinate_type<Policy0, Policy1>, Dims>>
      out;
  out.element = result.element;
  out.info.status = result.info.status;
  out.info.t = result.info.t;
  if (result) {
    out.info.point = ray.origin + result.info.t * ray.direction;
  }
  return out;
}

} // namespace tf

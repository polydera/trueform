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
#include "../core/frame_of.hpp"
#include "../core/ray_cast.hpp"
#include "../core/ray_like.hpp"
#include "../core/transformed.hpp"
#include "../core/form.hpp"
#include "./policy/tree.hpp"
#include "./tree/ray_cast_aabb.hpp"
#include "./tree/ray_cast_obb.hpp"
#include "./tree/ray_cast_obbrss.hpp"
#include "./tree_search/tree_ray_result.hpp"

namespace tf {

/// @ingroup spatial_queries
/// @brief Cast a ray against a form and find the first intersection.
///
/// Traverses the spatial tree to find the first primitive hit by the ray.
///
/// @param ray The ray to cast.
/// @param form The spatial form to query.
/// @param config Optional ray configuration (min/max t values).
/// @return A @ref tf::tree_ray_info containing the hit primitive ID and
///         intersection parameter t, or invalid if no hit.
template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray, const tf::form<Dims, Policy1> &form,
    tf::ray_config<tf::coordinate_type<Policy0, Policy1>> config = {}) {
  static_assert(tf::has_tree_policy<Policy1>,
                "Form must have a tree policy attached. Use: form | tf::tag(tree)");
  using tree_policy = typename Policy1::tree_policy;
  using Index = typename tree_policy::index_type;
  using real_t = tf::coordinate_type<Policy0, Policy1>;
  using bv_type = typename tree_policy::bv_type;

  auto l_ray = tf::transformed(ray, tf::inverse_transformation_of(form));

  tf::spatial::tree_ray_result<Index, tf::ray_cast_info<real_t>> result{
      config.min_t, config.max_t};

  tf::spatial::ray_cast(
      form.tree(), l_ray, result,
      [&form](const auto &l_ray, auto id) { return ray_cast(l_ray, form[id]); },
      bv_type{});

  return result.info();
}

} // namespace tf

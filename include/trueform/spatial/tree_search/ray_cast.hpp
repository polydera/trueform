/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/ray_cast.hpp"
#include "../../core/ray_like.hpp"
#include "../../core/transformed.hpp"
#include "../form.hpp"
#include "../tree/ray_cast_aabb.hpp"
#include "../tree/ray_cast_obb.hpp"
#include "../tree/ray_cast_obbrss.hpp"
#include "./tree_ray_result.hpp"

namespace tf {

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray, const tf::form<Dims, Policy1> &form,
    tf::ray_config<tf::coordinate_type<Policy0, Policy1>> config = {}) {
  using tree_policy = typename Policy1::tree_policy;
  using Index = typename tree_policy::index_type;
  using real_t = tf::coordinate_type<Policy0, Policy1>;
  using bv_type = typename tree_policy::bv_type;

  auto l_ray = tf::transformed(ray, form.inverse_transformation());

  tf::spatial::tree_ray_result<Index, tf::ray_cast_info<real_t>> result{
      config.min_t, config.max_t};

  tf::spatial::ray_cast(
      form.tree(), l_ray, result,
      [&form](const auto &l_ray, auto id) { return ray_cast(l_ray, form[id]); },
      bv_type{});

  return result.info();
}

} // namespace tf

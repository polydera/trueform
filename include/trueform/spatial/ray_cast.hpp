/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./form.hpp"
#include "../core/ray_cast.hpp"
#include "./search/tree_ray_cast.hpp"
#include "./search/tree_ray_info.hpp"
#include "../core/transformed.hpp"
#include "./tree.hpp"

namespace tf {
template <std::size_t Dims, typename Policy, typename Index, typename RealT,
          typename F>
auto ray_cast(
    const ray_like<Dims, Policy> &ray, const tf::tree<Index, RealT, Dims> &tree,
    const F &ray_cast_f,
    const tf::ray_config<tf::coordinate_type<Policy, RealT>> &config = {}) {
  tf::spatial::tree_ray_info<
      Index, tf::ray_cast_info<tf::coordinate_type<Policy, RealT>>>
      result{config.min_t, config.max_t};
  tf::spatial::tree_ray_cast(tree, ray, result, ray_cast_f);
  return result.info();
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_cast(
    const ray_like<Dims, Policy0> &ray, const tf::form<Dims, Policy1> &form,
    tf::ray_config<tf::coordinate_type<Policy0, Policy1>> config = {}) {
  auto l_ray = tf::transformed(ray, form.inverse_transformation());
  return ray_cast(
      l_ray, form.tree(),
      [&form](const auto &l_ray, auto id) { return ray_cast(l_ray, form[id]); },
      config);
}

} // namespace tf

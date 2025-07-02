/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/ray_hit_info.hpp"
#include "./ray_cast.hpp"

namespace tf {
template <std::size_t Dims, typename Policy, typename Index, typename RealT,
          typename F>
auto ray_hit(
    const ray_like<Dims, Policy> &ray, const tf::tree<Index, RealT, Dims> &tree,
    const F &ray_hit_f,
    const tf::ray_config<tf::coordinate_type<Policy, RealT>> &config = {}) {
  auto result = ray_cast(ray, tree, ray_hit_f, config);

  tf::ray_hit_info<tf::coordinate_type<Policy, RealT>, Dims> out;
  out.status = result.status;
  out.t = result.t;
  if (result) {
    out.point = ray.origin + result.t * ray.direction;
  }
  return out;
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto ray_hit(
    const ray_like<Dims, Policy0> &ray, const tf::form<Dims, Policy1> &form,
    tf::ray_config<tf::coordinate_type<Policy0, Policy1>> config = {}) {
  auto result = ray_cast(ray, form, config);

  tf::ray_hit_info<tf::coordinate_type<Policy0, Policy1>, Dims> out;
  out.status = result.status;
  out.t = result.t;
  if (result) {
    out.point = ray.origin + result.t * ray.direction;
  }
  return out;
}

} // namespace tf

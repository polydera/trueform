/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/ray_hit_info.hpp"
#include "./ray_cast.hpp"

namespace tf {

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

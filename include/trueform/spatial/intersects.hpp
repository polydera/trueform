/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/intersects.hpp"
#include "./ray_cast.hpp"
#include "./search.hpp"

namespace tf {

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::form<Dims, Policy0> &form0,
                const tf::form<Dims, Policy1> &form1) -> bool {
  return tf::search(form0, form1, tf::intersects_f, tf::intersects_f);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::point_like<Dims, Policy1> &obj) -> bool {
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj); },
      tf::intersects_f);
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::polygon<Dims, Policy1> &obj) -> bool {
  auto obj_aabb = tf::aabb_from(obj);
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj_aabb); },
      [&](const auto &other) { return tf::intersects(other, obj); });
}

template <std::size_t Dims, typename Policy0, typename Policy1>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::segment<Dims, Policy1> &obj) -> bool {
  auto ray = tf::make_ray_between_points(obj[0], obj[1]);
  using real_t = tf::coordinate_type<decltype(ray.origin)>;
  return tf::ray_cast(ray, form, tf::make_ray_config(real_t(0), real_t(1)));
}

template <std::size_t Dims, typename Policy0, typename Policy>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::ray_like<Dims, Policy> &obj) -> bool {
  return tf::ray_cast(obj, form);
}

template <std::size_t Dims, typename Policy0, typename Policy>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::line_like<Dims, Policy> &obj) -> bool {
  auto ray = tf::make_ray(obj.origin, obj.direction);
  using real_t = tf::coordinate_type<Policy0, Policy>;
  return tf::ray_cast(ray, form,
                      tf::make_ray_config(-std::numeric_limits<real_t>::max(),
                                          std::numeric_limits<real_t>::max()));
}

} // namespace tf

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/intersects.hpp"
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
      [&](const auto &other) { return tf::intersects(other, obj); });
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
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj); },
      [&](const auto &other) { return tf::intersects(other, obj); });
}

template <std::size_t Dims, typename Policy0, typename Policy>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::ray_like<Dims, Policy> &obj) -> bool {
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj); },
      [&](const auto &other) { return tf::intersects(other, obj); });
}

template <std::size_t Dims, typename Policy0, typename Policy>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::line_like<Dims, Policy> &obj) -> bool {
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj); },
      [&](const auto &other) { return tf::intersects(other, obj); });
}

template <std::size_t Dims, typename Policy0, typename Policy>
auto intersects(const tf::form<Dims, Policy0> &form,
                const tf::plane_like<Dims, Policy> &obj) -> bool {
  return tf::search(
      form, [&](const auto &aabb) { return intersects(aabb, obj); },
      [&](const auto &other) { return tf::intersects(other, obj); });
}

} // namespace tf

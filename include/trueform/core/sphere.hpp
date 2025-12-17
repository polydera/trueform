/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./base/sphere.hpp"
#include "./point_like.hpp"
#include "./sphere_like.hpp"

namespace tf {
template <typename T, std::size_t Dims>
using sphere =
    tf::sphere_like<Dims, tf::core::sphere<Dims, tf::core::pt<T, Dims>>>;

template <std::size_t Dims, typename Policy>
auto make_sphere(const point_like<Dims, Policy> &origin,
                 tf::coordinate_type<Policy> r) {
  return sphere<tf::coordinate_type<Policy>, Dims>{origin, r};
}
} // namespace tf

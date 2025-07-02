/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../coordinate_type.hpp"
#include "../point_like.hpp"

namespace tf::core {
template <std::size_t Dims, typename Policy> struct sphere {
  using coordinate_type = tf::coordinate_type<Policy>;

  sphere() = default;
  sphere(const tf::point_like<Dims, Policy> &origin, coordinate_type r)
      : origin{origin}, r{r} {}

  template <typename Policy1>
  auto operator=(const sphere<Dims, Policy1> &other) -> std::enable_if_t<
      std::is_assignable_v<tf::point_like<Dims, Policy> &,
                           tf::point_like<Dims, Policy1>> &&
          std::is_assignable_v<coordinate_type &,
                               typename Policy1::coordinate_type>,
      sphere &> {
    origin = other.origin;
    r = other.r;
    return *this;
  }

  tf::point_like<Dims, Policy> origin;
  coordinate_type r;
};
template <std::size_t Dims, typename Policy>
auto make_sphere(const point_like<Dims, Policy> &origin,
                 tf::coordinate_type<Policy> r) {
  return sphere<Dims, Policy>{origin, r};
}
} // namespace tf::core

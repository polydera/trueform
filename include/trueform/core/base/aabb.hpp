/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../coordinate_type.hpp"
#include "../point_like.hpp"
#include <type_traits>

namespace tf::core {
template <std::size_t Dims, typename Policy0, typename Policy1> struct aabb {
  static_assert(std::is_same_v<tf::coordinate_type<Policy0>,
                               tf::coordinate_type<Policy1>>);
  using coordinate_type = tf::coordinate_type<Policy0>;

  aabb() = default;
  aabb(const tf::point_like<Dims, Policy0> &min,
       const tf::point_like<Dims, Policy1> &max)
      : min{min}, max{max} {}

  tf::point_like<Dims, Policy0> min;
  tf::point_like<Dims, Policy0> max;
};

template <std::size_t N, typename T0, typename T1>
auto make_aabb(const point_like<N, T0> &min, const point_like<N, T1> &max)
    -> aabb<N, T0, T1> {
  return aabb<N, T0, T1>(min, max);
}
} // namespace tf::core

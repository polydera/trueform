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
#include "../coordinate_type.hpp"
#include "../point_like.hpp"
#include <type_traits>

namespace tf::core {
template <std::size_t Dims, typename Policy0, typename Policy1> struct aabb {
  static_assert(std::is_same_v<tf::coordinate_type<Policy0>,
                               tf::coordinate_type<Policy1>>);
  using coordinate_type = tf::coordinate_type<Policy0>;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  aabb() = default;
  aabb(const tf::point_like<Dims, Policy0> &min,
       const tf::point_like<Dims, Policy1> &max)
      : min{min}, max{max} {}

  template <typename Policy2, typename Policy3>
  auto operator=(const aabb<Dims, Policy2, Policy3> &other) -> std::enable_if_t<
      std::is_assignable_v<tf::point_like<Dims, Policy0> &,
                           tf::point_like<Dims, Policy2>> &&
          std::is_assignable_v<tf::point_like<Dims, Policy1> &,
                               tf::point_like<Dims, Policy3>>,
      aabb &> {
    min = other.min;
    max = other.max;
    return *this;
  }

  tf::point_like<Dims, Policy0> min;
  tf::point_like<Dims, Policy0> max;
};

template <std::size_t N, typename T0, typename T1>
auto make_aabb(const point_like<N, T0> &min, const point_like<N, T1> &max)
    -> aabb<N, T0, T1> {
  return aabb<N, T0, T1>(min, max);
}
} // namespace tf::core

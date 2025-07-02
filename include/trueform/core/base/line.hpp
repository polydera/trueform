/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../coordinate_type.hpp"
#include "../point_like.hpp"
#include "../vector_like.hpp"
#include <type_traits>

namespace tf::core {
template <std::size_t Dims, typename Policy0, typename Policy1> struct line {
  static_assert(std::is_same_v<tf::coordinate_type<Policy0>,
                               tf::coordinate_type<Policy1>>);
  using coordinate_type = tf::coordinate_type<Policy0>;

  line() = default;
  line(const tf::point_like<Dims, Policy0> &origin,
       const tf::vector_like<Dims, Policy1> &direction)
      : origin{origin}, direction{direction} {}

  template <typename Policy2, typename Policy3>
  auto operator=(const line<Dims, Policy2, Policy3> &other) -> std::enable_if_t<
      std::is_assignable_v<tf::point_like<Dims, Policy0> &,
                           tf::point_like<Dims, Policy2>> &&
          std::is_assignable_v<tf::vector_like<Dims, Policy1> &,
                               tf::vector_like<Dims, Policy3>>,
      line &> {
    origin = other.origin;
    direction = other.direction;
    return *this;
  }

  tf::point_like<Dims, Policy0> origin;
  tf::vector_like<Dims, Policy1> direction;
};

template <std::size_t N, typename T0, typename T1>
auto make_line(const point_like<N, T0> &origin,
               const vector_like<N, T1> &direction) -> line<N, T0, T1> {
  return line<N, T0, T1>(origin, direction);
}
} // namespace tf::core

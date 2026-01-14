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
#include "../unit_vector_like.hpp"
#include <type_traits>

namespace tf::core {
template <std::size_t Dims, typename Policy0, typename Policy1> struct rss {
  static_assert(std::is_same_v<tf::coordinate_type<Policy0>,
                               tf::coordinate_type<Policy1>>);
  using coordinate_type = tf::coordinate_type<Policy0>;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  rss() = default;
  rss(const tf::point_like<Dims, Policy0> &origin,
      const std::array<tf::unit_vector_like<Dims, Policy1>, Dims> &axes,
      const std::array<coordinate_type, Dims - 1> &length,
      coordinate_type radius)
      : origin{origin}, axes{axes}, length{length}, radius{radius} {}

  template <typename Policy2, typename Policy3>
  auto operator=(const rss<Dims, Policy2, Policy3> &other) -> std::enable_if_t<
      std::is_assignable_v<tf::point_like<Dims, Policy0> &,
                           tf::point_like<Dims, Policy2>> &&
          std::is_assignable_v<
              std::array<tf::unit_vector_like<Dims, Policy1>, Dims> &,
              std::array<tf::unit_vector_like<Dims, Policy3>, Dims>>,
      rss &> {
    origin = other.origin;
    axes = other.axes;
    length = other.length;
    radius = other.radius;
    return *this;
  }

  /// @brief Corner origin point.
  tf::point_like<Dims, Policy0> origin;

  /// @brief Orthonormal unit axes.
  std::array<tf::unit_vector_like<Dims, Policy1>, Dims> axes;

  /// @brief Rectangle side lengths along axes[0] through axes[Dims-2].
  std::array<coordinate_type, Dims - 1> length;

  /// @brief Sphere radius for the RSS capsule.
  coordinate_type radius;
};

template <std::size_t N, typename T0, typename T1>
auto make_rss(const point_like<N, T0> &origin,
              const std::array<unit_vector_like<N, T1>, N> &axes,
              const std::array<tf::coordinate_type<T0>, N - 1> &length,
              tf::coordinate_type<T0> radius)
    -> rss<N, T0, T1> {
  return rss<N, T0, T1>(origin, axes, length, radius);
}
} // namespace tf::core

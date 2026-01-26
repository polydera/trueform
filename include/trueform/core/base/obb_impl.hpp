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
template <std::size_t Dims, typename Policy0, typename Policy1> struct obb {
  static_assert(std::is_same_v<tf::coordinate_type<Policy0>,
                               tf::coordinate_type<Policy1>>);
  using coordinate_type = tf::coordinate_type<Policy0>;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  obb() = default;
  obb(const tf::point_like<Dims, Policy0> &origin,
      const std::array<tf::unit_vector_like<Dims, Policy1>, Dims> &axes,
      const std::array<coordinate_type, Dims> &extent)
      : origin{origin}, axes{axes}, extent{extent} {}

  template <typename Policy2, typename Policy3>
  auto operator=(const obb<Dims, Policy2, Policy3> &other) -> std::enable_if_t<
      std::is_assignable_v<tf::point_like<Dims, Policy0> &,
                           tf::point_like<Dims, Policy2>> &&
          std::is_assignable_v<
              std::array<tf::unit_vector_like<Dims, Policy1>, Dims> &,
              std::array<tf::unit_vector_like<Dims, Policy3>, Dims>>,
      obb &> {
    origin = other.origin;
    axes = other.axes;
    extent = other.extent;
    return *this;
  }

  /// @brief Corner origin point.
  tf::point_like<Dims, Policy0> origin;

  /// @brief Orthonormal unit axes.
  std::array<tf::unit_vector_like<Dims, Policy1>, Dims> axes;

  /// @brief Full extents along each axis.
  std::array<coordinate_type, Dims> extent;
};

template <std::size_t N, typename T0, typename T1>
auto make_obb(const point_like<N, T0> &origin,
              const std::array<unit_vector_like<N, T1>, N> &axes,
              const std::array<tf::coordinate_type<T0>, N> &extent)
    -> obb<N, T0, T1> {
  return obb<N, T0, T1>(origin, axes, extent);
}
} // namespace tf::core

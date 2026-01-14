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

/// @brief Combined OBB and RSS bounding volume.
///
/// This structure combines an Oriented Bounding Box (OBB) and a Rectangular
/// Swept Sphere (RSS) into a single structure. They share axes but have
/// separate origins because the RSS rectangle is shrunk and z-centered
/// relative to the OBB.
///
/// @tparam Dims The spatial dimension (e.g., 3 for 3D).
/// @tparam Policy0 The policy type for the origin point.
/// @tparam Policy1 The policy type for the axis vectors.
template <std::size_t Dims, typename Policy0, typename Policy1> struct obbrss {
  static_assert(std::is_same_v<tf::coordinate_type<Policy0>,
                               tf::coordinate_type<Policy1>>);
  using coordinate_type = tf::coordinate_type<Policy0>;
  using coordinate_dims = std::integral_constant<std::size_t, Dims>;

  obbrss() = default;
  obbrss(const tf::point_like<Dims, Policy0> &obb_origin,
         const tf::point_like<Dims, Policy0> &rss_origin,
         const std::array<tf::unit_vector_like<Dims, Policy1>, Dims> &axes,
         const std::array<coordinate_type, Dims> &extent,
         const std::array<coordinate_type, Dims - 1> &length,
         coordinate_type radius)
      : obb_origin{obb_origin}, rss_origin{rss_origin}, axes{axes},
        extent{extent}, length{length}, radius{radius} {}

  template <typename Policy2, typename Policy3>
  auto operator=(const obbrss<Dims, Policy2, Policy3> &other)
      -> std::enable_if_t<
          std::is_assignable_v<tf::point_like<Dims, Policy0> &,
                               tf::point_like<Dims, Policy2>> &&
              std::is_assignable_v<
                  std::array<tf::unit_vector_like<Dims, Policy1>, Dims> &,
                  std::array<tf::unit_vector_like<Dims, Policy3>, Dims>>,
          obbrss &> {
    obb_origin = other.obb_origin;
    rss_origin = other.rss_origin;
    axes = other.axes;
    extent = other.extent;
    length = other.length;
    radius = other.radius;
    return *this;
  }

  /// @brief Corner origin point for the OBB.
  tf::point_like<Dims, Policy0> obb_origin;

  /// @brief Corner origin point for the RSS rectangle (shrunk, z-centered).
  tf::point_like<Dims, Policy0> rss_origin;

  /// @brief Orthonormal unit axes (shared by OBB and RSS).
  std::array<tf::unit_vector_like<Dims, Policy1>, Dims> axes;

  /// @brief Full extents along each axis (OBB).
  std::array<coordinate_type, Dims> extent;

  /// @brief Rectangle side lengths along axes[0..Dims-2] (RSS).
  std::array<coordinate_type, Dims - 1> length;

  /// @brief Sphere radius for the RSS capsule.
  coordinate_type radius;
};

template <std::size_t N, typename T0, typename T1>
auto make_obbrss(const point_like<N, T0> &obb_origin,
                 const point_like<N, T0> &rss_origin,
                 const std::array<unit_vector_like<N, T1>, N> &axes,
                 const std::array<tf::coordinate_type<T0>, N> &extent,
                 const std::array<tf::coordinate_type<T0>, N - 1> &length,
                 tf::coordinate_type<T0> radius) -> obbrss<N, T0, T1> {
  return obbrss<N, T0, T1>(obb_origin, rss_origin, axes, extent, length, radius);
}

} // namespace tf::core

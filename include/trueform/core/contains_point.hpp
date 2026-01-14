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
#include "./contains_coplanar_point.hpp"
#include "./epsilon.hpp"
#include "./point_like.hpp"
#include "./policy/plane.hpp"
#include "./polygon.hpp"
#include <cstddef>

namespace tf {
namespace core {
/// @ingroup core_queries
/// @brief Checks whether a point lies inside a 2D polygon.
/// @tparam V Number of vertices in the polygon.
/// @tparam Policy Polygon storage policy.
/// @tparam T Coordinate type of the point.
/// @param poly The 2D polygon.
/// @param input_pt The 2D point to test.
/// @return True if the point lies inside the polygon.
template <typename Policy, typename T>
auto contains_point(const tf::polygon<2, Policy> &poly,
                    const point_like<2, T> &input_pt) -> containment {
  return core::contains_coplanar_point(
      poly, input_pt, tf::make_identity_projector(),
      tf::epsilon<tf::coordinate_type<Policy, T>>);
}
/// @ingroup core_queries
/// @brief Checks whether a point lies inside a 3D polygon by projecting to 2D.
/// @tparam V Number of vertices in the polygon.
/// @tparam Policy Polygon storage policy.
/// @tparam Dims Spatial dimensions.
/// @tparam T Coordinate type of the point.
/// @param poly The polygon.
/// @param input_pt The point to test.
/// @return True if the point lies inside the polygon.
template <std::size_t Dims, typename Policy, typename T>
auto contains_point(const tf::polygon<Dims, Policy> &poly_in,
                    const point_like<Dims, T> &input_pt) -> containment {
  const auto &poly = tf::tag_plane(poly_in);
  auto d = tf::dot(poly.plane().normal, input_pt) + poly.plane().d;
  if (std::abs(d) > tf::epsilon<decltype(d)>)
    return containment::outside;
  return core::contains_coplanar_point(
      poly, input_pt - d * poly.plane().normal,
      tf::make_simple_projector(poly.plane().normal), tf::epsilon<decltype(d)>);
}
} // namespace core
template <std::size_t Dims, typename Policy, typename T>
auto contains_point(const tf::polygon<Dims, Policy> &poly,
                    const point_like<Dims, T> &input_pt) -> bool {
  return core::contains_point(poly, input_pt) != containment::outside;
}
} // namespace tf

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
#include "./aabb.hpp"
#include "./aabb_union.hpp"
#include "./algorithm/reduce.hpp"
#include "./coordinate_type.hpp"
#include "./empty_aabb.hpp"
#include "./point_like.hpp"
#include "./points.hpp"
#include "./polygon.hpp"
#include "./polygons.hpp"
#include "./segment.hpp"
#include "./segments.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Construct an AABB from an aabb
///
/// The resulting AABB has identical `min` and `max` bounds equal to the input.
///
/// @tparam T The point policy
/// @tparam N The number of dimensions (e.g., 2 or 3).
/// @param pt The input point.
/// @return An axis-aligned bounding box
template <std::size_t N, typename T>
auto aabb_from(const aabb_like<N, T> &box) -> aabb<coordinate_type<T>, N> {
  return box;
}

/// @ingroup core_primitives
/// @brief Construct an AABB from a single point.
///
/// The resulting AABB has identical `min` and `max` bounds equal to the input
/// point.
///
/// @tparam T The point policy
/// @tparam N The number of dimensions (e.g., 2 or 3).
/// @param pt The input point.
/// @return An axis-aligned bounding box with zero extent at the point.
template <std::size_t N, typename T>
auto aabb_from(const point_like<N, T> &pt) -> aabb<tf::coordinate_type<T>, N> {
  return aabb<tf::coordinate_type<T>, N>{pt, pt};
}

/// @ingroup core_primitives
/// @brief Constructs an axis-aligned bounding box (AABB) from a polygon.
///
/// This overload computes the @ref tf::aabb that tightly encloses all points in
/// the input @ref tf::polygon. The bounding box is computed by iteratively
/// expanding the initial AABB (from the first vertex) to include each
/// subsequent point.
///
/// @tparam V The number of vertices in the polygon (may be `tf::dynamic_size`).
/// @tparam Policy The underlying point access policy.
/// @param poly The input polygon.
/// @return The minimal @ref tf::aabb that contains all vertices of the polygon.
template <std::size_t Dims, typename Policy>
auto aabb_from(const polygon<Dims, Policy> &poly) {
  auto out = aabb_from(poly[0]);
  for (std::size_t i = 1; i < std::size_t(poly.size()); ++i)
    aabb_union_inplace(out, poly[i]);
  return out;
}

/// @ingroup core_primitives
/// @brief Constructs an axis-aligned bounding box (AABB) from a segment.
///
/// This overload computes the @ref tf::aabb that tightly encloses the two
/// endpoints of the given @ref tf::segment. The result is the minimal bounding
/// box containing both points.
///
/// @tparam Policy The underlying point access policy.
/// @param s The input segment.
/// @return The minimal @ref tf::aabb enclosing both endpoints of the segment.
template <std::size_t Dims, typename Policy>
auto aabb_from(const segment<Dims, Policy> &s) {
  return aabb_union(aabb_from(s[0]), s[1]);
}

template <typename Policy> auto aabb_from(const tf::points<Policy> &points) {
  if (!points.size())
    return tf::make_empty_aabb<tf::coordinate_type<Policy>,
                               tf::coordinate_dims_v<Policy>>();
  auto out = tf::aabb_from(points.front());
  return tf::reduce(
      points, [](const auto &x, const auto &y) { return tf::aabb_union(x, y); },
      out, tf::checked);
}

template <typename Policy>
auto aabb_from(const tf::polygons<Policy> &polygons) {
  return aabb_from(polygons.points());
}

template <typename Policy>
auto aabb_from(const tf::segments<Policy> &segments) {
  return aabb_from(segments.points());
}

} // namespace tf

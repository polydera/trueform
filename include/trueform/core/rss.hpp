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

#include "./base/rss_impl.hpp"
#include "./coordinate_type.hpp"
#include "./rss_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Rectangle Swept Sphere in N-dimensional space.
///
/// Represents a rectangle (defined by origin, two axes, and their lengths)
/// swept by a sphere of given radius. The origin is the local (0,0,0) corner.
///
/// @tparam T The scalar type of the coordinates (e.g., float or double).
/// @tparam Dims The spatial dimension (e.g., 2 or 3).
template <typename T, std::size_t Dims>
using rss = tf::rss_like<
    Dims, tf::core::rss<Dims, tf::core::pt<T, Dims>, tf::core::vec<T, Dims>>>;

/// @ingroup core_primitives
/// @brief Construct an RSS from origin, axes, length, and radius.
///
/// A convenience function equivalent to directly calling the `rss<T, N>`
/// constructor.
///
/// @tparam N The spatial dimension.
/// @tparam T0 The point policy
/// @tparam T1 The vector policy
/// @param origin The local (0,0,0) corner of the rectangle.
/// @param axes The orientation unit axes.
/// @param length The lengths along the first N-1 axes.
/// @param radius The sphere radius.
/// @return An `rss<T, N>` instance.
template <std::size_t N, typename T0, typename T1>
auto make_rss(const point_like<N, T0> &origin,
              const std::array<unit_vector_like<N, T1>, N> &axes,
              const std::array<tf::coordinate_type<T0, T1>, N - 1> &length,
              tf::coordinate_type<T0, T1> radius)
    -> rss<tf::coordinate_type<T0, T1>, N> {
  return rss<tf::coordinate_type<T0, T1>, N>(origin, axes, length, radius);
}

} // namespace tf

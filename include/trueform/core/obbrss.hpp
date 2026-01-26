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

#include "./base/obbrss_impl.hpp"
#include "./coordinate_type.hpp"
#include "./obbrss_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Combined OBB and RSS bounding volume in N-dimensional space.
///
/// Represents an oriented bounding box combined with a rectangle swept sphere.
/// They share axes but have separate origins because the RSS rectangle is
/// shrunk and z-centered relative to the OBB.
///
/// @tparam T The scalar type of the coordinates (e.g., float or double).
/// @tparam Dims The spatial dimension (e.g., 2 or 3).
template <typename T, std::size_t Dims>
using obbrss = tf::obbrss_like<
    Dims,
    tf::core::obbrss<Dims, tf::core::pt<T, Dims>, tf::core::vec<T, Dims>>>;

/// @ingroup core_primitives
/// @brief Construct an OBBRSS from origins, axes, extent, length, and radius.
///
/// A convenience function equivalent to directly calling the `obbrss<T, N>`
/// constructor.
///
/// @tparam N The spatial dimension.
/// @tparam T0 The point policy
/// @tparam T1 The vector policy
/// @param obb_origin The corner origin for the OBB.
/// @param rss_origin The corner origin for the RSS rectangle (shrunk, z-centered).
/// @param axes The orientation unit axes.
/// @param extent The full extents along each axis (OBB).
/// @param length The lengths along the first N-1 axes (RSS rectangle).
/// @param radius The sphere radius (RSS).
/// @return An `obbrss<T, N>` instance.
template <std::size_t N, typename T0, typename T1>
auto make_obbrss(const point_like<N, T0> &obb_origin,
                 const point_like<N, T0> &rss_origin,
                 const std::array<unit_vector_like<N, T1>, N> &axes,
                 const std::array<tf::coordinate_type<T0, T1>, N> &extent,
                 const std::array<tf::coordinate_type<T0, T1>, N - 1> &length,
                 tf::coordinate_type<T0, T1> radius)
    -> obbrss<tf::coordinate_type<T0, T1>, N> {
  return obbrss<tf::coordinate_type<T0, T1>, N>(obb_origin, rss_origin, axes,
                                                 extent, length, radius);
}

} // namespace tf

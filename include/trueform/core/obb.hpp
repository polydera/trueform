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

#include "./base/obb_impl.hpp"
#include "./coordinate_type.hpp"
#include "./obb_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Oriented bounding box in N-dimensional space.
///
/// Represents an oriented rectangular region defined by its origin (local
/// (0,0,0) corner), axes (orientation), and extent (full dimensions along each axis).
///
/// @tparam T The scalar type of the coordinates (e.g., float or double).
/// @tparam Dims The spatial dimension (e.g., 2 or 3).
template <typename T, std::size_t Dims>
using obb = tf::obb_like<
    Dims, tf::core::obb<Dims, tf::core::pt<T, Dims>, tf::core::vec<T, Dims>>>;

/// @ingroup core_primitives
/// @brief Construct an OBB from origin, axes, and extent.
///
/// A convenience function equivalent to directly calling the `obb<T, N>`
/// constructor.
///
/// @tparam N The spatial dimension.
/// @tparam T0 The point policy
/// @tparam T1 The vector policy
/// @param origin The local (0,0,0) corner of the bounding box.
/// @param axes The orientation unit axes.
/// @param extent The full extents along each axis.
/// @return An `obb<T, N>` instance.
template <std::size_t N, typename T0, typename T1>
auto make_obb(const point_like<N, T0> &origin,
              const std::array<unit_vector_like<N, T1>, N> &axes,
              const std::array<tf::coordinate_type<T0, T1>, N> &extent)
    -> obb<tf::coordinate_type<T0, T1>, N> {
  return obb<tf::coordinate_type<T0, T1>, N>(origin, axes, extent);
}

} // namespace tf

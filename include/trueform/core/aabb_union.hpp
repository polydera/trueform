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
#include "./aabb_like.hpp"
#include "./coordinate_type.hpp"
#include "./point_like.hpp"

namespace tf {

/// @ingroup core_primitives
/// @brief Expand an AABB to include another AABB, in-place.
///
/// Updates `aabb0` to include the bounds of `aabb1`. Modifies `aabb0` directly.
///
/// @tparam T The scalar coordinate type.
/// @tparam N The spatial dimension.
/// @param aabb0 The AABB to be expanded.
/// @param aabb1 The AABB to include.
/// @return A reference to `aabb0`.
template <std::size_t N, typename T0, typename T1>
auto aabb_union_inplace(aabb_like<N, T0> &aabb0, const aabb_like<N, T1> &aabb1)
    -> aabb_like<N, T0> & {
  for (int i = 0; i < int(N); i++) {
    aabb0.min[i] = std::min(aabb0.min[i], coordinate_type<T0>(aabb1.min[i]));
    aabb0.max[i] = std::max(aabb0.max[i], coordinate_type<T0>(aabb1.max[i]));
  }
  return aabb0;
}

/// @ingroup core_primitives
/// @brief Compute the union of two AABBs.
///
/// Returns a new AABB that bounds both `aabb0` and `aabb1`.
/// Neither input is modified.
///
/// @tparam T The scalar coordinate type.
/// @tparam N The spatial dimension.
/// @param aabb0 The first bounding box.
/// @param aabb1 The second bounding box.
/// @return An `aabb<T, N>` containing both inputs.
template <std::size_t N, typename T0, typename T1>
auto aabb_union(const aabb_like<N, T0> &aabb0, const aabb_like<N, T1> &aabb1) {
  aabb<tf::coordinate_type<T0, T1>, N> out = aabb0;
  aabb_union_inplace(out, aabb1);
  return out;
}

/// @ingroup core_primitives
/// @brief Expand an AABB to include a point, in-place.
///
/// Updates `aabb0` to include the given point `pt`.
///
/// @tparam T The scalar coordinate type.
/// @tparam T1 The point policy
/// @tparam N The spatial dimension.
/// @param aabb0 The AABB to be expanded.
/// @param pt The point to include.
/// @return A reference to `aabb0`.
template <std::size_t N, typename T0, typename T1>
auto aabb_union_inplace(aabb_like<N, T0> &aabb0, const point_like<N, T1> &pt)
    -> aabb_like<N, T0> & {
  for (int i = 0; i < int(N); i++) {
    aabb0.min[i] = std::min(aabb0.min[i], coordinate_type<T0>(pt[i]));
    aabb0.max[i] = std::max(aabb0.max[i], coordinate_type<T0>(pt[i]));
  }
  return aabb0;
}

/// @ingroup core_primitives
/// @brief Compute the union of an AABB and a point.
///
/// Returns a new AABB that includes both the input bounding box and the given
/// point. The original AABB is not modified.
///
/// @tparam T The scalar coordinate type.
/// @tparam N The spatial dimension.
/// @tparam T1 The point policy
/// @param aabb0 The bounding box.
/// @param pt The point to include.
/// @return An `aabb<T, N>` containing the original box and the point.
template <std::size_t N, typename T0, typename T1>
auto aabb_union(const aabb_like<N, T0> &aabb0, const point_like<N, T1> &pt) {
  aabb<tf::coordinate_type<T0, T1>, N> out = aabb0;
  aabb_union_inplace(out, pt);
  return out;
}
} // namespace tf

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

namespace tf {

/// @ingroup core_primitives
/// @brief Return a copy of an AABB inflated on every axis on both sides.
///
/// `min` is moved down by `by`, `max` is moved up by `by`. `by == 0`
/// returns the AABB unchanged.
///
/// @tparam Dims The spatial dimension.
/// @tparam Policy The AABB policy.
/// @param a The AABB to inflate.
/// @param by The per-axis inflation, in the AABB's coordinate units.
/// @return A new `aabb<T, Dims>` with the inflated extents.
template <std::size_t Dims, typename Policy>
auto inflated_aabb(const tf::aabb_like<Dims, Policy> &a,
                   tf::coordinate_type<Policy> by)
    -> tf::aabb<tf::coordinate_type<Policy>, Dims> {
  tf::aabb<tf::coordinate_type<Policy>, Dims> out = tf::make_aabb(a.min, a.max);
  if (by == tf::coordinate_type<Policy>(0))
    return out;
  for (std::size_t k = 0; k < Dims; ++k) {
    out.min[k] -= by;
    out.max[k] += by;
  }
  return out;
}

} // namespace tf

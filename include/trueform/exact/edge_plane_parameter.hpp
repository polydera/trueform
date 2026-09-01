/*
 * Copyright (c) 2026 XLAB
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

#include "./edge_parameter.hpp"
#include "./orient3d.hpp"

namespace tf::exact {

/// The exact parameter of segment (p0, p1)'s crossing with the plane
/// of (a, b, c): the same orient3d volumes
/// @ref tf::exact::segment_plane_intersect snaps, kept as a fraction.
/// An endpoint on the plane yields the exact 0 or 1; a crossing beyond
/// the endpoints yields its exact line position. Caller ensures the
/// segment is not parallel to the plane (equal volumes at both ends).
template <typename Int>
auto make_edge_plane_parameter(const pt3<Int> &a, const pt3<Int> &b,
                               const pt3<Int> &c, const pt3<Int> &p0,
                               const pt3<Int> &p1) -> edge_parameter<Int> {
  auto vol_0 = orient3d_value(a, b, c, p0);
  auto vol_1 = orient3d_value(a, b, c, p1);
  auto num = vol_0;
  auto den = vol_0 - vol_1;
  if (den < decltype(den)(0)) {
    num = -num;
    den = -den;
  }
  return {num, den};
}

} // namespace tf::exact

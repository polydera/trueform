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

/// orient3d sign of the point at parameter `t` along (p0, p1) against
/// the plane of (a, b, c) — without materializing the point. Zero is
/// an exact incidence: the junction predicate. The volume at `t`
/// scaled by the (positive) denominator is
/// den * vol(p0) + num * (vol(p1) - vol(p0)).
template <typename Int>
auto parameter_plane_sign(const pt3<Int> &a, const pt3<Int> &b,
                          const pt3<Int> &c, const pt3<Int> &p0,
                          const pt3<Int> &p1, const edge_parameter<Int> &t)
    -> int {
  auto vol_0 = orient3d_value(a, b, c, p0);
  auto vol_1 = orient3d_value(a, b, c, p1);
  return det2_sign<Int>(t.den, vol_0, t.num, vol_0 - vol_1);
}

} // namespace tf::exact

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

#include "./det2_sign.hpp"
#include "./edge_parameter.hpp"
#include "./orient2d.hpp"

namespace tf::exact {

/// orient2d sign of the point at parameter `t` along (p0, p1) against
/// the directed line (a, b) — without materializing the point. Zero is
/// an exact incidence. The orientation at `t` scaled by the (positive)
/// denominator is den * O(p0) + num * (O(p1) - O(p0)) where
/// O(x) = orient2d(a, b, x).
template <typename Int>
auto parameter_line_sign2(const pt2<Int> &a, const pt2<Int> &b,
                          const pt2<Int> &p0, const pt2<Int> &p1,
                          const edge_parameter<Int> &t) -> int {
  auto o_0 = orient2d(a, b, p0);
  auto o_1 = orient2d(a, b, p1);
  return det2_sign<Int>(t.den, o_0, t.num, o_0 - o_1);
}

} // namespace tf::exact

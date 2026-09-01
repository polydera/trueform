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

#include "./orient2d.hpp"

namespace tf::exact {

/// Proper segment crossing test: do segments (a,b) and (c,d) properly cross?
/// Returns true only for proper crossings (not touching at endpoints or
/// collinear overlap).
template <typename Int>
auto segments_cross(const pt2<Int> &a, const pt2<Int> &b, const pt2<Int> &c,
                    const pt2<Int> &d) -> bool {
  auto o1 = orient2d(a, b, c);
  auto o2 = orient2d(a, b, d);
  auto o3 = orient2d(c, d, a);
  auto o4 = orient2d(c, d, b);
  if (((o1 > 0 && o2 < 0) || (o1 < 0 && o2 > 0)) &&
      ((o3 > 0 && o4 < 0) || (o3 < 0 && o4 > 0)))
    return true;
  return false;
}

} // namespace tf::exact

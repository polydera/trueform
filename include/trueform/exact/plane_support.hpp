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

#include "../core/point.hpp"
#include "./meta.hpp"

#include <array>

namespace tf::exact {

/// THE ONE PRODUCER of the points a carrier stands on: the first point
/// offered to it, the first offered point distinct from that one, and the
/// first offered point off their line. Points arrive in the caller's own
/// order, so the plane they state is the carrier's own, and the cross the
/// search must compute to reject a collinear candidate IS that plane's
/// exact normal.
///
/// Fewer than three points is the verdict itself: two state the LINE the
/// carrier collapsed to, one states its single point, and the normal stays
/// zero.
template <typename Int> struct plane_support {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  std::array<tf::point<Int, 3>, 3> point{};
  /// `point[1] - point[0]`, the direction a two-point support states
  std::array<T1, 3> edge{};
  /// `edge x (point[2] - point[0])`, zero until the support is complete
  std::array<T2, 3> normal{};
  int size = 0;

  /// True when the offered point joined the support. A complete support
  /// takes nothing more, so an offering loop may stop at `size == 3`, and a
  /// pull-shaped one names each member by the index it offered it from.
  auto offer(const tf::point<Int, 3> &p) -> bool {
    if (size == 0) {
      point[0] = p;
      size = 1;
      return true;
    }
    if (size == 3)
      return false;
    const std::array<T1, 3> from_first{T1(p[0]) - point[0][0],
                                       T1(p[1]) - point[0][1],
                                       T1(p[2]) - point[0][2]};
    if (size == 1) {
      if (from_first[0] == T1(0) && from_first[1] == T1(0) &&
          from_first[2] == T1(0))
        return false;
      point[1] = p;
      edge = from_first;
      size = 2;
      return true;
    }
    const std::array<T2, 3> cross{
        T2(edge[1]) * from_first[2] - T2(edge[2]) * from_first[1],
        T2(edge[2]) * from_first[0] - T2(edge[0]) * from_first[2],
        T2(edge[0]) * from_first[1] - T2(edge[1]) * from_first[0]};
    if (cross[0] == T2(0) && cross[1] == T2(0) && cross[2] == T2(0))
      return false;
    point[2] = p;
    normal = cross;
    size = 3;
    return true;
  }
};

} // namespace tf::exact

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

#include "./meta.hpp"
#include "./plane_frame.hpp"
#include "./plane_support.hpp"
#include "./projection_axes.hpp"

#include <cstddef>

namespace tf::exact {

/// THE ONE PRODUCER of a carrier's exact frame: the reading of the points
/// the carrier stands on (@ref tf::exact::plane_support), as
/// @ref tf::exact::make_canonical_plane is the reading of its name, so a
/// carrier's frame and its name are one verdict. A complete support decides
/// the plane. A support of two points IS A LINE: its normal is zero, and its
/// axes are the pair holding the LINE'S OWN dominant direction, so the
/// projection stays injective on the carrier and a coincidence in the
/// projection IS a coincidence in space. One point is a point; nothing
/// separates it, so its axes are `{0, 1}`.
template <typename Int>
auto make_supported_plane_frame(const tf::exact::plane_support<Int> &support)
    -> tf::exact::plane_frame<Int> {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  tf::exact::plane_frame<Int> frame;
  if (support.size == 3) {
    const auto axes = tf::exact::projection_axes<Int>(support.normal);
    frame.ax0 = int(axes.first);
    frame.ax1 = int(axes.second);
    frame.plane_n = support.normal;
    frame.plane_d = frame.plane_n[0] * T2(support.point[0][0]) +
                    frame.plane_n[1] * T2(support.point[0][1]) +
                    frame.plane_n[2] * T2(support.point[0][2]);
  } else if (support.size == 2) {
    const auto magnitude = [](T1 value) {
      return value < T1(0) ? -value : value;
    };
    std::size_t dominant = 0;
    if (magnitude(support.edge[1]) > magnitude(support.edge[0]))
      dominant = 1;
    if (magnitude(support.edge[2]) > magnitude(support.edge[dominant]))
      dominant = 2;
    frame.ax0 = dominant == 2 ? 0 : int(dominant);
    frame.ax1 = dominant == 2 ? 2 : int(dominant) + 1;
  }
  frame.plane_fallback = support.point[0][3 - frame.ax0 - frame.ax1];
  return frame;
}

} // namespace tf::exact

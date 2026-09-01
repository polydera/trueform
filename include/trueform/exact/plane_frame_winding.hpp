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

#include "./plane_frame.hpp"
#include "./signed_area.hpp"
#include "./vertex.hpp"

#include <cstdint>

namespace tf::exact {

/// A loop's winding in the frame's projection, exact: `+1` when the
/// projected loop turns positively, `-1` when it turns the other way. The
/// answer is a side, so a loop of no area takes the positive one rather
/// than a third value no consumer can act on.
template <typename Int, typename Loop, typename GetPoint>
auto plane_frame_winding(const tf::exact::plane_frame<Int> &frame,
                         const Loop &loop, const GetPoint &get_point)
    -> std::int8_t {
  return tf::exact::signed_area_2x(loop,
                                   [&](auto corner) -> tf::exact::pt2<Int> {
                                     const auto q = get_point(corner);
                                     return {q[frame.ax0], q[frame.ax1]};
                                   }) >= 0
             ? std::int8_t(1)
             : std::int8_t(-1);
}

} // namespace tf::exact

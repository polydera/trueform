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

#include "../../../core/point.hpp"
#include "../../canonical_plane.hpp"
#include "../../plane_support.hpp"

namespace tf::exact::door::pool {

/// The EXACT plane a face names: the canonical quadruple through the
/// three original lattice vertices its own support stands on. It is the
/// same currency the downstream namer speaks, so an elected
/// representative is a plane some face actually stood on and not one the
/// grid invented.
///
/// False for a face with no support triple, which names no plane.
template <typename Int, typename Corners>
auto make_support_plane(const Corners &corners,
                        tf::exact::canonical_plane<Int> &plane) -> bool {
  tf::exact::plane_support<Int> support;
  for (const auto &corner : corners) {
    if (support.size == 3)
      break;
    support.offer(tf::point<Int, 3>{corner[0], corner[1], corner[2]});
  }
  if (support.size < 3)
    return false;
  plane = tf::exact::make_canonical_plane<Int>(support);
  return true;
}

} // namespace tf::exact::door::pool

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
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Halve coordinates before measuring spans so the full signed range fits.
template <typename Coord> struct tet_site_domain {
  static auto half(Coord c) -> Coord { return c >> 1U; }

  auto offset(std::size_t axis, Coord c) const -> Coord {
    return half(c) - minimum[axis];
  }

  /// The domain holding both this one and `other`.
  auto merged(const tet_site_domain &other) const -> tet_site_domain {
    tet_site_domain out;
    for (std::size_t axis = 0; axis < 3; ++axis) {
      out.minimum[axis] = std::min(minimum[axis], other.minimum[axis]);
      out.span[axis] = std::max(minimum[axis] + span[axis],
                                other.minimum[axis] + other.span[axis]) -
                       out.minimum[axis];
    }
    return out;
  }

  std::array<Coord, 3> minimum;
  std::array<Coord, 3> span;
};

} // namespace tf::topology::cdt::dt3

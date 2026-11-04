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
#include "../../../../exact/vertex.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// The point at `numerator / 2^bits` along the segment from `a` to `b`, the
/// one place a split's coordinate is computed: exact, in `Wide`, read `bits`
/// deeper than its ends are.
template <typename Wide, typename Point>
auto tet_segment_point(const Point &a, const Point &b, std::int64_t numerator,
                       int bits) -> tf::exact::pt3<Wide> {
  tf::exact::pt3<Wide> p;
  for (std::size_t axis = 0; axis < 3; ++axis)
    p[axis] = Wide(a[axis]) * (Wide(1) << bits) +
              (Wide(b[axis]) - Wide(a[axis])) * Wide(numerator);
  return p;
}

} // namespace tf::topology::cdt::dt3

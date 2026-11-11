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
#include "./tet_segment_recovery.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// Deepen the frame of `segment` by one bit: every numerator its rows and
/// the rings at its ends state doubles, so each names the same point on the
/// frame twice as fine, and a row with no numerator inside it has one.
template <typename Index, typename Coord>
auto deepen_tet_segment_frame(tet_segment_recovery<Index, Coord> &recovery,
                              Index segment) -> void {
  auto &rows = recovery.subsegments;
  const auto twice = [](std::int64_t &numerator) {
    if (numerator >= 0)
      numerator *= 2;
  };
  ++recovery.bits[std::size_t(segment)];
  twice(recovery.rings[2 * std::size_t(segment)]);
  twice(recovery.rings[2 * std::size_t(segment) + 1]);
  for (Index k = recovery.first[std::size_t(segment)]; k != Index(-1);
       k = rows[std::size_t(k)].next) {
    twice(rows[std::size_t(k)].n0);
    twice(rows[std::size_t(k)].n1);
  }
}

} // namespace tf::topology::cdt::dt3

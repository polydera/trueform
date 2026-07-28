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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index, typename DeadLoops, typename CoplanarPairs>
auto apply_region_stack_aliases(
    const DeadLoops &dead_loops, const CoplanarPairs &coplanar_pairs,
    tf::buffer<std::array<Index, 2>> &ranges, tf::buffer<char> &reversed,
    bool &aliased) -> void {
  reversed.allocate(ranges.size());
  tf::parallel_fill(reversed, char(0));
  for (const auto &pair : coplanar_pairs) {
    if (dead_loops.size() != 0 &&
        bool(dead_loops[std::size_t(pair[0])]))
      continue;
    ranges[std::size_t(pair[1])] = ranges[std::size_t(pair[0])];
    reversed[std::size_t(pair[1])] = char(pair[2]);
    aliased = true;
  }
}

} // namespace tf::cut

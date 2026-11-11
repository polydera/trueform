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
#include <array>
#include <cstddef>

namespace tf::topology {

/// Whether a loop touches itself: some vertex carries more than two of its
/// passes. A loop that does not is a simple cycle, and its winding is its
/// parity everywhere. `pass(i)`, for each of `n_passes` walls, names its two
/// vertices and how many times the loop runs it, zero for a wall it does not
/// run; `tally(vertex, passes)` adds those passes at a vertex and answers its
/// count so far.
template <typename Pass, typename Tally>
auto loop_touches_itself(std::size_t n_passes, const Pass &pass,
                         const Tally &tally) -> bool {
  for (std::size_t i = 0; i < n_passes; ++i) {
    const auto wall = pass(i);
    if (wall[2] == 0)
      continue;
    for (int end = 0; end < 2; ++end)
      if (tally(wall[std::size_t(end)], wall[2]) > 2)
        return true;
  }
  return false;
}

} // namespace tf::topology

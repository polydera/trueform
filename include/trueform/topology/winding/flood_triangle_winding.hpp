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
#include "../../core/buffer.hpp"
#include <algorithm>
#include <climits>
#include <cstddef>

namespace tf::topology {

/// The winding of a loop around every one of `n_triangles` triangles: the
/// exterior winds nothing, a triangle with a side nothing stands across —
/// `neighbor(t, e)` of `-1` — starts at the step entering it there, and the
/// winding moves across every other side by `entering(t, e)`, the step a wall
/// adds entering `t` across its side `e`, zero across a side no wall is.
/// `stack` is the caller's; `winding` is stated for every triangle the flood
/// reaches.
template <typename Index, typename Neighbor, typename Entering>
auto flood_triangle_winding(std::size_t n_triangles, const Neighbor &neighbor,
                            const Entering &entering, tf::buffer<Index> &stack,
                            tf::buffer<signed char> &winding) -> void {
  const auto unset = static_cast<signed char>(SCHAR_MIN);
  winding.allocate(n_triangles);
  std::fill(winding.begin(), winding.end(), unset);
  stack.clear();
  for (std::size_t triangle = 0; triangle < n_triangles; ++triangle)
    for (int edge = 0; edge < 3; ++edge)
      if (neighbor(Index(triangle), edge) == Index(-1)) {
        winding[triangle] =
            static_cast<signed char>(entering(Index(triangle), edge));
        stack.push_back(Index(triangle));
        break;
      }
  while (stack.size() != 0) {
    const auto triangle = stack[stack.size() - 1];
    stack.pop_back();
    for (int edge = 0; edge < 3; ++edge) {
      const auto next = neighbor(triangle, edge);
      if (next == Index(-1) || winding[std::size_t(next)] != unset)
        continue;
      winding[std::size_t(next)] = static_cast<signed char>(
          winding[std::size_t(triangle)] - entering(triangle, edge));
      stack.push_back(next);
    }
  }
}

} // namespace tf::topology

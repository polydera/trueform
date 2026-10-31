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
#include "./laplacian_system.hpp"
#include <cstddef>

namespace tf::geometry {

/// Whether every connected run of free rows reaches a fixed one.
///
/// Where every weight is positive, that is exactly what makes the form's free
/// block definite without a tether: `x^T H_FF x` vanishes only where `[x; 0]`
/// is constant on each component, so only where a component holds no fixed
/// row. The proof is the walk, taken before any factor is, so an unanchored
/// run is refused rather than left to a roundoff-sized pivot.
template <typename Index>
auto laplacian_is_anchored(const tf::geometry::laplacian_system<Index> &system,
                           const tf::buffer<Index> &free_ids,
                           const tf::buffer<Index> &free_of,
                           tf::buffer<Index> &stack, tf::buffer<char> &seen)
    -> bool {
  seen.allocate_and_initialize(free_ids.size(), char(0));
  for (std::size_t k = 0; k < free_ids.size(); ++k) {
    if (seen[k])
      continue;
    bool anchored = false;
    stack.clear();
    stack.push_back(Index(k));
    seen[k] = char(1);
    while (stack.size()) {
      const std::size_t slot = std::size_t(stack.back());
      stack.pop_back();
      const std::size_t row = std::size_t(free_ids[slot]);
      for (Index at = system.row_offsets[row]; at < system.row_offsets[row + 1];
           ++at) {
        const Index peer =
            free_of[std::size_t(system.columns[std::size_t(at)])];
        if (std::size_t(peer) >= free_ids.size()) {
          anchored = true;
          continue;
        }
        if (seen[std::size_t(peer)])
          continue;
        seen[std::size_t(peer)] = char(1);
        stack.push_back(peer);
      }
    }
    if (!anchored)
      return false;
  }
  return true;
}

} // namespace tf::geometry

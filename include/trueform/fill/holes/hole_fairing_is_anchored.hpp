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
#include "./hole_fairing_system.hpp"
#include <cstddef>

namespace tf::fill {

/// Whether every connected run of free vertices reaches a fixed one, which is
/// what makes the form definite on them without a tether.
template <typename Index>
auto hole_fairing_is_anchored(
    const tf::fill::hole_fairing_system<Index> &system,
    const tf::buffer<Index> &free_ids, const tf::buffer<Index> &free_of,
    tf::buffer<Index> &stack, tf::buffer<char> &seen) -> bool {
  seen.allocate(free_ids.size());
  for (std::size_t k = 0; k < free_ids.size(); ++k)
    seen[k] = char(0);
  for (std::size_t k = 0; k < free_ids.size(); ++k) {
    if (seen[k])
      continue;
    bool anchored = false;
    stack.clear();
    stack.push_back(Index(k));
    seen[k] = char(1);
    while (stack.size()) {
      const std::size_t slot = std::size_t(stack.back());
      stack.erase_till_end(stack.end() - 1);
      const std::size_t row = std::size_t(free_ids[slot]);
      for (Index at = system.row_offsets[row]; at < system.row_offsets[row + 1];
           ++at) {
        const Index column = system.columns[std::size_t(at)];
        const Index peer = free_of[std::size_t(column)];
        if (peer < 0) {
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

} // namespace tf::fill

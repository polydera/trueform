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
#include "../cholesky_factor.hpp"
#include "../cholesky_pattern.hpp"
#include "./cholesky_front_scratch.hpp"
#include "./factor_supernode.hpp"
#include <cstddef>

namespace tf::linalg {

/// Factor the supernodes `schedule[from..to)`, ascending, on one stack.
///
/// `feed(s)` runs before supernode `s` and pushes the updates of any of its
/// children the run does not factor itself, in ascending order; in a
/// postorder a supernode's children are then the stack's top entries, the
/// ones whose parent it is. Returns false at the first pivot that is not
/// positive or not finite.
template <typename Index, typename Feed>
auto factor_supernode_run(const tf::cholesky_pattern<Index> &pattern,
                          const tf::buffer<double> &lower_values,
                          std::size_t from, std::size_t to, Feed &&feed,
                          tf::linalg::cholesky_front_scratch<Index> &scratch,
                          tf::cholesky_factor &factor) -> bool {
  scratch.position.allocate(pattern.size());
  scratch.stack.clear();
  scratch.pending.clear();
  scratch.pending_offsets.clear();
  for (std::size_t slot = from; slot < to; ++slot) {
    const std::size_t s = std::size_t(pattern.schedule[slot]);
    feed(s);
    std::size_t children = 0;
    while (children < scratch.pending.size() &&
           std::size_t(pattern.supernode_parent[std::size_t(
               scratch.pending[scratch.pending.size() - 1 - children])]) == s)
      ++children;
    if (!tf::linalg::factor_supernode(pattern, lower_values, s, children,
                                      scratch, factor))
      return false;
  }
  return true;
}

} // namespace tf::linalg

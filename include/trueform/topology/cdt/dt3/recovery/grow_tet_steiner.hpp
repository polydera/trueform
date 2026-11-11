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
#include "../../../../core/algorithm/parallel_fill.hpp"
#include "../../../../core/views/drop.hpp"
#include "./tet_segment_recovery.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Grow the recovery's `steiner` mask to `n_sites` slots, the new slots
/// minted by no tier of the recovery until a mint marks its own. O(the new
/// slots).
template <typename Index, typename Coord>
auto grow_tet_steiner(tet_segment_recovery<Index, Coord> &recovery,
                      std::size_t n_sites) -> void {
  auto &steiner = recovery.steiner;
  const std::size_t known = steiner.size();
  if (n_sites <= known)
    return;
  steiner.reallocate(n_sites);
  tf::parallel_fill(tf::drop(steiner, known), char(0));
}

} // namespace tf::topology::cdt::dt3

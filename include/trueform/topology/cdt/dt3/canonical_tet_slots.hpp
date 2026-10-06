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
#include <utility>

namespace tf::topology::cdt::dt3 {

/// The slot order a finite cell is published in: corners ascending, the last
/// two exchanged when that order is an odd permutation of the stored one, so
/// the published cell keeps the stored orientation.
template <typename Corners>
auto canonical_tet_slots(const Corners &corners) -> std::array<std::size_t, 4> {
  std::array<std::size_t, 4> slots{0, 1, 2, 3};
  bool odd = false;
  for (std::size_t i = 1; i < 4; ++i)
    for (std::size_t j = i; j > 0 && corners[slots[j]] < corners[slots[j - 1]];
         --j) {
      std::swap(slots[j], slots[j - 1]);
      odd = !odd;
    }
  if (odd)
    std::swap(slots[2], slots[3]);
  return slots;
}

} // namespace tf::topology::cdt::dt3

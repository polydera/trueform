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
#include <cstdint>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// The slot order a finite cell is published in, packed two bits a slot:
/// corners ascending, the last two exchanged when that order is an odd
/// permutation of the stored one, so the published cell keeps the stored
/// orientation. `ascending` receives the corners in ascending order, the
/// key the cell is ranked by. Five compare-exchanges sort four slots.
template <typename Corners, typename Index>
auto canonical_tet_slots(const Corners &corners,
                         std::array<Index, 4> &ascending) -> std::uint8_t {
  std::array<std::uint8_t, 4> slots{0, 1, 2, 3};
  bool odd = false;
  const auto order = [&corners, &slots, &odd](std::size_t a, std::size_t b) {
    if (corners[std::size_t(slots[b])] < corners[std::size_t(slots[a])]) {
      std::swap(slots[a], slots[b]);
      odd = !odd;
    }
  };
  order(0, 1);
  order(2, 3);
  order(0, 2);
  order(1, 3);
  order(1, 2);
  for (std::size_t i = 0; i < 4; ++i)
    ascending[i] = Index(corners[std::size_t(slots[i])]);
  if (odd)
    std::swap(slots[2], slots[3]);
  return std::uint8_t(slots[0] | slots[1] << 2 | slots[2] << 4 | slots[3] << 6);
}

/// Slot `i` of a packed slot order.
inline constexpr auto canonical_tet_slot(std::uint8_t slots, std::size_t i)
    -> std::size_t {
  return std::size_t((slots >> (2 * i)) & 3U);
}

} // namespace tf::topology::cdt::dt3

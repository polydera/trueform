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
#include "./tet_cell_key.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// The key a finite cell is ranked and published by: its corners ascending
/// and the slot order it is published in, packed two bits a slot — the
/// ascending order, the last two exchanged when that order is an odd
/// permutation of the stored one, so the published cell keeps the stored
/// orientation. Five compare-exchanges sort four slots, each a select.
template <typename Corners, typename Index>
auto canonical_tet_key(const Corners &corners, Index row)
    -> tet_cell_key<Index> {
  std::array<Index, 4> values{Index(corners[0]), Index(corners[1]),
                              Index(corners[2]), Index(corners[3])};
  std::array<unsigned, 4> slots{0, 1, 2, 3};
  unsigned odd = 0;
  const auto order = [&values, &slots, &odd](std::size_t a, std::size_t b) {
    const bool exchange = values[b] < values[a];
    const Index low = exchange ? values[b] : values[a];
    const Index high = exchange ? values[a] : values[b];
    const unsigned first = exchange ? slots[b] : slots[a];
    const unsigned second = exchange ? slots[a] : slots[b];
    values[a] = low;
    values[b] = high;
    slots[a] = first;
    slots[b] = second;
    odd ^= unsigned(exchange);
  };
  order(0, 1);
  order(2, 3);
  order(0, 2);
  order(1, 3);
  order(1, 2);
  const unsigned third = odd ? slots[3] : slots[2];
  const unsigned fourth = odd ? slots[2] : slots[3];
  return {values, row,
          std::uint8_t(slots[0] | slots[1] << 2 | third << 4 | fourth << 6),
          odd != 0};
}

/// The key a hull cell is ranked and published by: its three finite corners
/// ascending and the infinite one last, and the slot order it is published
/// in, packed two bits a slot — its finite facet stated from its smallest
/// corner in the stored cyclic order, the infinite corner in slot three, so
/// the published cell keeps the stored orientation; that order exchanges the
/// second and third ascending corners where the stored cycle descends.
template <typename Corners, typename Index>
auto canonical_tet_hull_key(const Corners &corners, Index row)
    -> tet_cell_key<Index> {
  const Index a = corners[0], b = corners[1], c = corners[2];
  const unsigned lead = a < b ? (a < c ? 0U : 2U) : (b < c ? 1U : 2U);
  const unsigned second = lead == 2 ? 0U : lead + 1;
  const unsigned third = second == 2 ? 0U : second + 1;
  const Index x = corners[second], y = corners[third];
  const bool exchange = y < x;
  return {{Index(corners[lead]), exchange ? y : x, exchange ? x : y,
           Index(corners[3])},
          row,
          std::uint8_t(lead | second << 2 | third << 4 | 3U << 6),
          exchange};
}

/// Slot `i` of a packed slot order.
inline constexpr auto canonical_tet_slot(std::uint8_t slots, std::size_t i)
    -> std::size_t {
  return std::size_t((slots >> (2 * i)) & 3U);
}

} // namespace tf::topology::cdt::dt3

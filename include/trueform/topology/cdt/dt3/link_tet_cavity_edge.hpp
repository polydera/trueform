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
#include "./tet_cavity_edge_table.hpp"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

template <typename Owner, typename Index>
auto link_tet_cavity_edge(Owner &owner, tet_cavity_edge_table<Index> &table,
                          Index a, Index b, Index tet, int wall) -> void {
  const auto key = [a, b] {
    if constexpr (sizeof(Index) <= 4)
      return (std::uint64_t(std::uint32_t(a)) << 32) | std::uint32_t(b);
    else
      return std::array<Index, 2>{a, b};
  }();
  const auto hash = [key] {
    if constexpr (sizeof(Index) <= 4)
      return key;
    else
      return std::uint64_t(key[0]) * 0x9e3779b97f4a7c15ULL ^
             std::uint64_t(key[1]);
  }();
  using Table = tet_cavity_edge_table<Index>;
  auto bucket =
      std::size_t((hash * 11400714819323198485ULL) >> (64U - Table::bits));
  for (;;) {
    auto &slot = table.slots[bucket];
    if (slot.generation != table.generation) {
      slot = {key, Index(4 * std::size_t(tet) + std::size_t(wall)),
              table.generation};
      return;
    }
    if (slot.key == key) {
      assert(slot.ticket >= 0);
      const auto other = std::size_t(slot.ticket) / 4,
                 side = std::size_t(slot.ticket) % 4;
      owner._neighbors[std::size_t(tet)][std::size_t(wall)] = Index(other);
      owner._neighbors[other][side] = tet;
      slot.ticket = -1;
      return;
    }
    bucket = (bucket + 1) & (Table::size - 1);
  }
}

} // namespace tf::topology::cdt::dt3

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
#include <type_traits>

namespace tf::topology::cdt::dt3 {

template <typename Index> struct tet_cavity_edge_slot {
  std::conditional_t<(sizeof(Index) <= 4), std::uint64_t, std::array<Index, 2>>
      key{};
  Index ticket = -1;
  std::uint32_t generation = 0;
};

/// Open-addressed edge slots for one star, reset by advancing the generation.
template <typename Index> struct tet_cavity_edge_table {
  static constexpr unsigned bits = 9;
  static constexpr std::size_t size = std::size_t(1) << bits;
  /// A star of `w` walls states `3w / 2` distinct edges, so up to this many
  /// walls the table stays at most three-eighths full and every probe ends
  /// at a free slot; a larger star links its edges by sorting instead.
  static constexpr std::size_t max_walls = size / 4;
  std::array<tet_cavity_edge_slot<Index>, size> slots{};
  std::uint32_t generation = 0;
};

} // namespace tf::topology::cdt::dt3

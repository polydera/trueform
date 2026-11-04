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
#include "../../../core/buffer.hpp"
#include "./tet_edge_ref.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// One entry of a ring under flips (@ref flip_tet_ring): a live cell of the
/// ring as an edge, or, past the live prefix, the record of a flip that
/// shrank the ring — the 2-3 of the facet at `at` whose apex `apex` left the
/// link, the n-m flip of the link edge `pivot` (1 for c b, 2 for a c) of that
/// facet whose star of `n1` cells lies at `span` in the arena, or the 4-4 of
/// a flat ring of four, `span` and `n1` then the edge it made.
template <typename Index> struct tet_ring_entry {
  static constexpr std::uint8_t live = 0;
  static constexpr std::uint8_t facet_flip = 1;
  static constexpr std::uint8_t ring_flip = 2;
  static constexpr std::uint8_t quad_flip = 3;
  Index cell;
  std::uint8_t a;
  std::uint8_t b;
  std::uint8_t kind;
  std::uint8_t pivot;
  Index at;
  Index apex;
  Index span;
  Index n1;
};

/// The scratch every flip removal walks on: per row how many rings under
/// flips hold it; the arena the rings live in, a stack of spans; the cells a
/// removal made and the segments it met, each when asked; the edges held as
/// segments for one face's recovery; and the limits — the link level a
/// removal may reach, `-1` for the automatic level, which rises round by
/// round, and the ring size past which an edge is left alone, `-1` for none.
template <typename Index> struct tet_flip_scratch {
  tf::buffer<std::uint8_t> counts;
  tf::buffer<tet_ring_entry<Index>> arena;
  tf::buffer<tet_edge_ref<Index>> collected;
  tf::buffer<std::array<Index, 2>> encroached;
  int level_limit = -1;
  int auto_level = 1;
  int star_limit = -1;
  std::size_t hull_barred = 0;
};

/// Note the segment `p q` a removal met, once.
template <typename Index>
auto note_tet_encroached_segment(tet_flip_scratch<Index> &scratch, Index p,
                                 Index q) -> void {
  const std::array<Index, 2> key{p < q ? p : q, p < q ? q : p};
  for (const auto &known : scratch.encroached)
    if (known == key)
      return;
  scratch.encroached.push_back(key);
}

/// The rows a flip wrote are in no ring.
template <typename Index, typename Rows>
auto tet_ring_fresh(tet_flip_scratch<Index> &scratch, const Rows &rows)
    -> void {
  for (auto row : rows)
    tet_ring_count(scratch, row) = 0;
}

/// The rings holding `cell`; a row the lane does not reach is in none.
template <typename Index>
auto tet_ring_count(tet_flip_scratch<Index> &scratch, Index cell)
    -> std::uint8_t & {
  auto &counts = scratch.counts;
  if (std::size_t(cell) >= counts.size()) {
    const std::size_t old = counts.size();
    counts.reallocate(std::size_t(cell) + 1);
    for (std::size_t i = old; i < counts.size(); ++i)
      counts[i] = 0;
  }
  return counts[std::size_t(cell)];
}

template <typename Index>
auto tet_ring_count(const tet_flip_scratch<Index> &scratch, Index cell)
    -> std::uint8_t {
  const auto &counts = scratch.counts;
  return std::size_t(cell) < counts.size() ? counts[std::size_t(cell)] : 0;
}

/// The level a removal may reach now.
template <typename Index>
auto tet_flip_level_limit(const tet_flip_scratch<Index> &scratch) -> int {
  return scratch.level_limit < 0 ? scratch.auto_level : scratch.level_limit;
}

} // namespace tf::topology::cdt::dt3

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
#include "./recovery/tet_stuck_facet.hpp"
#include "./tet_edge_ref.hpp"
#include "./recovery/tet_star_stamps.hpp"
#include "./tet_edge_star.hpp"
#include "./tet_execution_tuning.hpp"
#include "./tet_round_state.hpp"
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

/// The scratch every flip removal walks on: the rows it touched the ring
/// counts of, so it can leave none standing (@ref tet_ring_count,
/// @ref tet_ring_release — the counts themselves are the complex's lane); the arena the rings live in, a stack of spans; the cells a removal
/// made, when asked; the edges held as segments for one face's recovery; and
/// the limits — the link level a removal may reach, `-1` for the automatic
/// level, which rises round by round, and the ring size past which an edge is
/// left alone, `-1` for none; `hull_barred`, the rings a hull cell kept
/// from flipping, counted across every removal since the segments were first
/// stated (@ref prepare_tet_segments); `tickets` the facets a restoration
/// sweep still owes and `stuck` the ones it could not flip, both the task's
/// own where a round runs the regions beside each other, and `star` with its
/// `star_stamps` the walk's own gather — EVERY SCRATCH A TASK TOUCHES IS
/// ITS REGION'S, a stamp being no row and so no admission's to guard;
/// `walk_passes` the
/// turns one walk of a segment may take and `walk_capped` the walks that
/// spent them, read as a delta by the tier that refuses the row; and
/// `round` what a
/// region's flips take and write while a round runs the regions beside each
/// other (@ref tet_round_state).
template <typename Index> struct tet_flip_scratch {
  tf::buffer<Index> touched;
  tf::buffer<tet_ring_entry<Index>> arena;
  tf::buffer<tet_edge_ref<Index>> collected;
  int level_limit = -1;
  int auto_level = 1;
  int star_limit = -1;
  std::size_t walk_passes = tet_execution_tuning::recovery_walk_passes;
  std::size_t walk_capped = 0;
  std::size_t hull_barred = 0;
  tf::buffer<Index> tickets;
  tf::buffer<tet_stuck_facet<Index>> stuck;
  tet_edge_star<Index> star;
  tet_star_stamps star_stamps;
  tet_round_state<Index> round;
};

/// The rings holding `cell`, to state: the row joins the removal's touched
/// list, so the removal can release every count it set
/// (@ref tet_ring_release). A row the lane does not reach is in no ring, and
/// a writer grows the lane to its row.
template <typename Owner>
auto tet_ring_count(Owner &owner,
                    tet_flip_scratch<typename Owner::index_type> &scratch,
                    typename Owner::index_type cell) -> std::uint8_t & {
  scratch.touched.push_back(cell);
  auto &counts = owner._ring_counts;
  if (std::size_t(cell) >= counts.size()) {
    const std::size_t old = counts.size();
    counts.reallocate(std::size_t(cell) + 1);
    for (std::size_t i = old; i < counts.size(); ++i)
      counts[i] = 0;
  }
  return counts[std::size_t(cell)];
}

/// The rings holding `cell`, to read.
template <typename Owner>
auto tet_ring_count(const Owner &owner, typename Owner::index_type cell)
    -> std::uint8_t {
  const auto &counts = owner._ring_counts;
  return std::size_t(cell) < counts.size() ? counts[std::size_t(cell)] : 0;
}

/// The rows a flip wrote are in no ring.
template <typename Owner, typename Rows>
auto tet_ring_fresh(Owner &owner,
                    tet_flip_scratch<typename Owner::index_type> &scratch,
                    const Rows &rows) -> void {
  for (auto row : rows)
    if (row >= 0)
      tet_ring_count(owner, scratch, row) = 0;
}

/// Release every count a removal set, from the mark it took at its start:
/// A REMOVAL LEAVES NO COUNT STANDING. The count says how many rings under
/// flips hold a row, and at a removal's end no ring holds anything; a
/// standing one is read by the next removal for a row it has taken over — a
/// fresh row, or any row once a reclamation renumbers the lanes — and the
/// flips' own decisions turn on it. O(the rows the removal touched), never
/// the complex's rows.
template <typename Owner>
auto tet_ring_release(Owner &owner,
                      tet_flip_scratch<typename Owner::index_type> &scratch,
                      std::size_t mark) -> void {
  auto &touched = scratch.touched;
  for (std::size_t at = mark; at < touched.size(); ++at)
    owner._ring_counts[std::size_t(touched[at])] = 0;
  touched.erase_till_end(touched.begin() + std::ptrdiff_t(mark));
}

/// The level a removal may reach now.
template <typename Index>
auto tet_flip_level_limit(const tet_flip_scratch<Index> &scratch) -> int {
  return scratch.level_limit < 0 ? scratch.auto_level : scratch.level_limit;
}

} // namespace tf::topology::cdt::dt3

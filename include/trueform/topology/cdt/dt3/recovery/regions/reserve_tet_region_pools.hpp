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
#include "../../../../../core/algorithm/parallel_fill.hpp"
#include "../../../../../core/algorithm/parallel_for_each.hpp"
#include "../../../../../core/buffer.hpp"
#include "../../../../../core/checked.hpp"
#include "../../../../../core/views/sequence_range.hpp"
#include "../../reserve_tet_protection.hpp"
#include "../../tet_row_pool.hpp"
#include "../../tet_state.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// The rows one region's pool holds per item due in it where the items are
/// flips, over the rows those flips can make, and the floor every region
/// holds besides: a 2-3 takes one row and the removals one item drives are
/// few. A round whose items MAKE NO ROW — the face phase's holds — asks for
/// none, and its pool is the one row that keeps it open.
inline constexpr std::size_t tet_region_pool_rows = 4;
inline constexpr std::size_t tet_region_pool_floor = 1024;

/// One pool per region, reserved past the complex's rows by prefix: the
/// region's own block, its rows dead until a flip takes one, so a fresh
/// row's index is a function of (region, order) and never of timing. The
/// ring lane and the protection lanes are sized to the rows AND the pools
/// here — a writer that grew one inside the round would move
/// it under the others' writes, which is the one hazard the admission cannot
/// answer, a lane being no row; every reserved row
/// marked for the region that holds it, so no writer grows it inside the
/// round. A pool row's own region is read from the first corner the flip
/// that takes it writes, so there is nothing to mark. O(due + regions) plus
/// O(the pool rows) to state them dead.
template <typename Owner, typename Index>
auto reserve_tet_region_pools(Owner &owner,
                              const tf::buffer<Index> &offsets,
                              std::size_t n_regions, std::size_t rows_per_item,
                              tf::buffer<tet_row_pool> &pools) -> void {
  pools.allocate(n_regions);
  const std::size_t standing = std::size_t(owner._corners.size());
  std::size_t at = standing;
  for (std::size_t p = 0; p < n_regions; ++p) {
    const std::size_t held =
        rows_per_item ? std::size_t(offsets[p + 1]) - std::size_t(offsets[p])
                      : 0;
    pools[p] = {at, at + held * rows_per_item +
                        (rows_per_item ? tet_region_pool_floor : 1)};
    at = pools[p].end;
  }
  owner._corners.reallocate(at);
  owner._neighbors.reallocate(at);
  owner._states.reallocate(at);
  // A RESERVED ROW IS STATED BY ITS STATE AND ITS FIRST CORNER AND NOTHING
  // MORE: dead, so every reader passes it over and the compaction takes it
  // back, and nobody's, so the admission answers for it (@ref
  // tet_row_admits). No link names it until a flip writes it whole, so its
  // other corners and its neighbours are the flip's to state — the
  // difference is 8 bytes a row against 36.
  tf::parallel_for_each(
      tf::make_sequence_range(standing, at),
      [&owner](std::size_t row) {
        owner._states[row] = tet_state::dead;
        owner._corners[row][0] = Owner::none;
      },
      tf::checked);
  owner._n_dead += at - standing;
  reserve_tet_protection(owner);
  {
    auto &lane = owner._ring_counts;
    const std::size_t known = lane.size();
    if (known < at) {
      lane.reallocate(at);
      tf::parallel_fill(
          tf::make_range(lane.begin() + std::ptrdiff_t(known), lane.end()),
          std::uint8_t(0));
    }
  }
}

} // namespace tf::topology::cdt::dt3

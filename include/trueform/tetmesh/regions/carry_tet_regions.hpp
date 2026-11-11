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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/reduce.hpp"
#include "../../core/algorithm/sequenced_generate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../topology/cdt/dt3/compact_tetrahedralization.hpp"
#include "../../topology/cdt/dt3/tet_facet_is_protected.hpp"
#include "../../topology/cdt/dt3/tet_state.hpp"
#include "./tet_regions.hpp"
#include <array>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace tf::tetmesh {

/// The winding of a row no closure has reached yet.
inline constexpr int unknown_winding = std::numeric_limits<int>::min();

/// Carry `regions`, stated over the first `n_rows` rows of `owner` and its
/// `n_finite` finite cells, across the flips written since, then publish the
/// complex canonically and compact the regions with it, so a reader sees the
/// cells by their rank and not by the restoration's writing order. A flip
/// removes no protected facet, so a region keeps the union of its cells and a
/// row a flip rewrote stays in its region; a row past `n_rows` lies in the
/// region of every cell it meets on an unprotected facet, and takes that
/// region and its winding from the first row below `n_rows` the closure over
/// those facets reaches. Returns the finite rows no closure reaches, those of
/// a region every row of which the flips replaced.
///
/// Every flip writes into at least two of the rows it reads, and only the
/// 3-2 flip retires one, so a region keeps a row below `n_rows` unless a
/// cascade of those retired its every original, and the region labeling states
/// the winding again where one does.
template <typename Owner>
auto carry_tet_regions(Owner &owner,
                       tet_regions<typename Owner::index_type> &regions,
                       std::size_t n_finite, std::size_t n_rows)
    -> std::size_t {
  using Index = typename Owner::index_type;
  using tf::topology::cdt::dt3::tet_state;
  const std::size_t rows = owner._corners.size();
  tf::buffer<int> winding;
  winding.allocate(rows);
  regions.regions.reallocate(rows);
  tf::parallel_for_each(
      tf::make_sequence_range(rows),
      [&winding, &regions, n_finite, n_rows](std::size_t row) {
        winding[row] = row < n_finite ? regions.winding[row]
                       : row < n_rows ? 0
                                      : unknown_winding;
      },
      tf::checked);
  // the rows past `n_rows` meeting a row below it seed the closure, in row
  // order; the closure itself is a walk
  tf::buffer<Index> queue;
  tf::sequenced_generate(
      tf::make_sequence_range(n_rows, rows), queue,
      [&owner, &winding, &regions, n_rows](std::size_t row, auto &out) {
        if (owner._states[row] == tet_state::dead)
          return;
        for (std::size_t slot = 0; slot < 4; ++slot) {
          const auto beyond = std::size_t(owner._neighbors[row][slot]);
          if (beyond >= n_rows ||
              tf::topology::cdt::dt3::tet_facet_is_protected(owner, Index(row),
                                                             slot))
            continue;
          winding[row] = winding[beyond];
          regions.regions[row] = regions.regions[beyond];
          out.push_back(Index(row));
          return;
        }
      },
      tf::checked);
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const auto row = std::size_t(queue[head]);
    for (std::size_t slot = 0; slot < 4; ++slot) {
      const auto beyond = std::size_t(owner._neighbors[row][slot]);
      if (beyond < n_rows || beyond >= rows ||
          winding[beyond] != unknown_winding ||
          tf::topology::cdt::dt3::tet_facet_is_protected(owner, Index(row),
                                                         slot))
        continue;
      winding[beyond] = winding[row];
      regions.regions[beyond] = regions.regions[row];
      queue.push_back(Index(beyond));
    }
  }
  using count_t = std::array<std::size_t, 1>;
  const std::size_t unreached = tf::reduce(
      tf::make_sequence_range(n_rows, rows),
      [&owner, &winding](count_t sum, const auto &at) {
        if constexpr (std::is_same_v<std::decay_t<decltype(at)>, count_t>)
          sum[0] += at[0];
        else
          sum[0] += owner._states[std::size_t(at)] == tet_state::live &&
                    winding[std::size_t(at)] == unknown_winding;
        return sum;
      },
      count_t{0}, tf::checked)[0];
  tf::topology::cdt::dt3::compact_tetrahedralization<true, true>(owner);
  const std::size_t n_kept = owner._n_finite;
  tf::buffer<Index> compacted;
  compacted.allocate(owner._corners.size());
  regions.winding.allocate(n_kept);
  tf::parallel_for_each(
      tf::make_sequence_range(rows),
      [&owner, &regions, &winding, &compacted, n_kept](std::size_t row) {
        const Index to = owner._compaction[row];
        if (to == Owner::none)
          return;
        compacted[std::size_t(to)] = regions.regions[row];
        if (std::size_t(to) < n_kept)
          regions.winding[std::size_t(to)] = winding[row];
      },
      tf::checked);
  std::swap(regions.regions, compacted);
  return unreached;
}

} // namespace tf::tetmesh

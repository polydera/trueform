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

#include "./line_cells.hpp"
#include "./make_line_plane.hpp"
#include "./pool_records.hpp"

#include "../../../core/algorithm/generic_generate.hpp"
#include "../../../core/algorithm/parallel_fill.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"

#include "tbb/parallel_sort.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::exact::door::pool {

/// The partition as a reader consumes it: one direct exact plane per
/// committed name.
///
/// THE PUBLISHED PLANE IS THE ONE THE CERTIFICATE JUDGED: the cell's elected
/// line through the winning anchor's own support vertex — a real direction
/// through a real lattice point, the same construction the offer was
/// certified against, so nothing rational the ladder computed reaches a
/// placement.
///
/// A name no anchor claimed keeps no plane at all, which is not the same
/// fact as being alone in a pool: an unclaimed name keeps whatever authority
/// it already had.
///
/// `owner_of_name` names the ANCHOR that claimed each name, so the plane a
/// pool publishes is read straight off the anchor and needs no side table
/// between them. Pools are numbered by their least member name, so the
/// numbering is a function of the geometry.
template <typename Int>
auto publish_line_pools(const pool_names<Int> &names,
                        const line_cells<Int> &cells,
                        const tf::buffer<int> &owner_of_name,
                        election_census &census, pool_partition<Int> &partition)
    -> void {
  const auto count = names.plane.size();
  partition.pool_of_name.allocate(count);
  partition.pool_plane.clear();
  tf::parallel_fill(partition.pool_of_name, -1);
  if (count == 0)
    return;

  // the least member of an owner's claim is a first occurrence over one
  // ascending walk, which is the sweep's own shape and stays serial
  tf::buffer<int> least_of_owner;
  least_of_owner.allocate(count);
  tf::parallel_fill(least_of_owner, -1);
  for (std::size_t n = 0; n < count; ++n) {
    const int owner = owner_of_name[n];
    if (owner >= 0 && least_of_owner[std::size_t(owner)] < 0)
      least_of_owner[std::size_t(owner)] = int(n);
  }

  // the pair is total — a pool's least member then a name — so the claimed
  // names may be gathered in any order
  tf::buffer<std::array<int, 2>> record;
  tf::generic_generate(
      tf::make_sequence_range(count), record,
      [&owner_of_name, &least_of_owner](
          std::size_t n, tf::buffer<std::array<int, 2>> &out) {
        const int owner = owner_of_name[n];
        if (owner >= 0)
          out.push_back({least_of_owner[std::size_t(owner)], int(n)});
      },
      tf::checked);
  tbb::parallel_sort(record.begin(), record.end());

  for (std::size_t k = 0; k < record.size(); ++k) {
    if (k == 0 || record[k][0] != record[k - 1][0]) {
      const auto anchor = std::size_t(owner_of_name[std::size_t(record[k][1])]);
      const auto cell = std::size_t(cells.cell_of_name[anchor]);
      partition.pool_plane.push_back(make_line_plane<Int>(
          names.plane[std::size_t(cells.elected[cell])],
          names.support_point[anchor]));
    }
    partition.pool_of_name[std::size_t(record[k][1])] =
        int(partition.pool_plane.size()) - 1;
  }

  census.pools = std::int64_t(partition.pool_plane.size());

#ifdef TF_POOL_CENSUS
  // the histogram exists for `exact_singletons` and nothing reads it below
  tf::buffer<int> pooled;
  pooled.allocate(partition.pool_plane.size());
  for (auto &at : pooled)
    at = 0;
  for (const auto pool : partition.pool_of_name) {
    if (pool < 0) {
      ++census.unpooled_names;
      continue;
    }
    ++census.pooled_names;
    ++pooled[std::size_t(pool)];
  }
  for (const auto at : pooled)
    census.exact_singletons += at == 1 ? 1 : 0;
#endif
}

} // namespace tf::exact::door::pool

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
#include "../../../../../core/buffer.hpp"
#include <algorithm>
#include "../../tet_execution_tuning.hpp"
#include "../../../../../core/views/sequence_range.hpp"
#include "../../../../../core/checked.hpp"
#include "../../../../../core/algorithm/parallel_for_each.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The due work bucketed by the region its anchor stands in: `order` holds
/// the POSITION in `due` of every item once, the positions of one region
/// contiguous and in `due`'s own order, and `offsets` the n_regions + 2
/// bounds of those runs, the last holding the items no region owns.
/// `patch_of(i)` gives the region of the item at position `i`.
///
/// Counts in parallel, one prefix, disjoint writes: the due range is cut
/// into blocks of a fixed length, every block counts its own items per
/// region, one prefix over the (block, region) counts gives each block its
/// base in each region's run, and every block then writes its own items
/// there. Both passes read the same blocks, so the order is a function of
/// `due` alone — a region's items come in `due`'s order at any worker count.
/// `cursor` is the caller's scratch and holds the counts.
template <typename Index, typename PatchOf>
auto bucket_tet_due_rows(std::size_t due, const PatchOf &patch_of,
                         std::size_t n_regions, tf::buffer<Index> &cursor,
                         tf::buffer<Index> &offsets, tf::buffer<Index> &order)
    -> void {
  const std::size_t buckets = n_regions + 1;
  const auto bucket = [&patch_of, n_regions](std::size_t at) -> std::size_t {
    const Index patch = patch_of(at);
    return patch < 0 || std::size_t(patch) >= n_regions ? n_regions
                                                        : std::size_t(patch);
  };
  const std::size_t blocks =
      (due + tet_execution_tuning::scan_block_rows - 1) /
      tet_execution_tuning::scan_block_rows;
  cursor.allocate_and_initialize(blocks * buckets + buckets, Index(0));
  tf::parallel_for_each(
      tf::make_sequence_range(blocks),
      [&cursor, &bucket, buckets, due](std::size_t block) {
        const std::size_t from = block * tet_execution_tuning::scan_block_rows;
        const std::size_t to =
            std::min(due, from + tet_execution_tuning::scan_block_rows);
        for (std::size_t at = from; at < to; ++at)
          ++cursor[block * buckets + bucket(at)];
      },
      tf::checked);
  offsets.allocate_and_initialize(buckets + 1, Index(0));
  {
    Index running = 0;
    for (std::size_t b = 0; b < buckets; ++b) {
      offsets[b] = running;
      for (std::size_t block = 0; block < blocks; ++block) {
        const Index held = cursor[block * buckets + b];
        cursor[block * buckets + b] = running;
        running += held;
      }
    }
    offsets[buckets] = running;
  }
  order.allocate(due);
  tf::parallel_for_each(
      tf::make_sequence_range(blocks),
      [&cursor, &order, &bucket, buckets, due](std::size_t block) {
        const std::size_t from = block * tet_execution_tuning::scan_block_rows;
        const std::size_t to =
            std::min(due, from + tet_execution_tuning::scan_block_rows);
        for (std::size_t at = from; at < to; ++at)
          order[std::size_t(cursor[block * buckets + bucket(at)]++)] =
              Index(at);
      },
      tf::checked);
}

} // namespace tf::topology::cdt::dt3

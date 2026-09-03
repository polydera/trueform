/*
 * Copyright (c) 2025 XLAB
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

#include "for_each_obj_partition.hpp"
#include "obj_corner_attributes.hpp"

#include "../../core/buffer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::io::obj {

/// @brief A corner and the position that buckets it.
template <typename Index> struct obj_bucketed_corner {
  int position;
  Index corner;
};

/// @brief Groups every corner's attribute record under the position it names.
///
/// A position is a dense bounded id, so the grouping is counts plus one prefix.
/// Counting straight into the positions would give each chunk a table as large
/// as the whole position space, so the positions are cut into one bucket per
/// chunk first: a chunk's counts are then `buckets` numbers, one prefix gives
/// every `(bucket, chunk)` pair a disjoint slice, and each bucket owns a
/// contiguous run of positions whose own counting sort runs independently and
/// inside cache.
///
/// @param corner_positions The position every corner names.
/// @param corner_attributes The `(texture, normal)` every corner names.
/// @param n_positions The size of the position id space.
/// @param chunk_count How many corner chunks carry the counting passes.
/// @param offsets Filled with each position's record start, of size
/// `n_positions + 1`.
/// @param records Filled with the corners, grouped by the position they name.
template <typename Index>
auto bucket_obj_corners_by_position(
    const tf::buffer<int> &corner_positions,
    const tf::buffer<std::array<int, 2>> &corner_attributes,
    std::size_t n_positions, std::size_t chunk_count,
    tf::buffer<Index> &offsets,
    tf::buffer<obj_corner_attributes<Index>> &records) -> void {
  const auto n_corners = corner_positions.size();
  std::size_t shift = 0;
  while ((n_positions >> shift) > chunk_count)
    ++shift;
  const auto bucket_count = ((n_positions - 1) >> shift) + 1;

  tf::buffer<Index> slices;
  slices.allocate_and_initialize(bucket_count * chunk_count + 1, Index(0));
  for_each_obj_partition(chunk_count, [&](std::size_t chunk) {
    const auto last = n_corners * (chunk + 1) / chunk_count;
    for (auto corner = n_corners * chunk / chunk_count; corner < last; ++corner)
      ++slices[(std::size_t(corner_positions[corner]) >> shift) * chunk_count +
               chunk];
  });
  Index filled = 0;
  for (std::size_t slice = 0; slice < bucket_count * chunk_count; ++slice) {
    filled += slices[slice];
    slices[slice] = filled;
  }
  slices[bucket_count * chunk_count] = filled;

  tf::buffer<obj_bucketed_corner<Index>> bucketed;
  bucketed.allocate(n_corners);
  for_each_obj_partition(chunk_count, [&](std::size_t chunk) {
    const auto last = n_corners * (chunk + 1) / chunk_count;
    for (auto corner = n_corners * chunk / chunk_count; corner < last;
         ++corner) {
      const auto position = corner_positions[corner];
      bucketed[--slices[(std::size_t(position) >> shift) * chunk_count +
                        chunk]] = {position, static_cast<Index>(corner)};
    }
  });

  offsets.allocate_and_initialize(n_positions + 1, Index(0));
  records.allocate(n_corners);
  for_each_obj_partition(bucket_count, [&](std::size_t bucket) {
    const auto first = slices[bucket * chunk_count];
    const auto last = slices[(bucket + 1) * chunk_count];
    for (auto slot = first; slot < last; ++slot)
      ++offsets[bucketed[slot].position];
    auto bucket_filled = first;
    const auto stop = std::min(n_positions, (bucket + 1) << shift);
    for (auto position = bucket << shift; position < stop; ++position) {
      bucket_filled += offsets[position];
      offsets[position] = bucket_filled;
    }
    for (auto slot = first; slot < last; ++slot) {
      const auto &entry = bucketed[slot];
      const auto &attributes = corner_attributes[entry.corner];
      records[--offsets[entry.position]] = {attributes[0], attributes[1],
                                            entry.corner};
    }
  });
  offsets[n_positions] = static_cast<Index>(n_corners);
}

} // namespace tf::io::obj

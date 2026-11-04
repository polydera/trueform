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
#include "../../../../core/algorithm/parallel_for_each.hpp"
#include "../../../../core/buffer.hpp"
#include "../../../../core/checked.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include <algorithm>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// The records one block of @ref bucket_tet_records counts and scatters.
inline constexpr std::size_t tet_record_block_size = std::size_t(1) << 16;

/// Order `records` by `key_of`, a key below `n_keys`, and each key's run by
/// `before`: counts over the records in blocks, one prefix into `offsets`
/// (block `k` of the result is key `k`'s run), each block scattering into
/// its own disjoint ranges of `scratch`, and the runs sorted apart in
/// parallel; `records` holds the result and `scratch` what it held.
template <typename Record, typename Index, typename KeyOf, typename Before>
auto bucket_tet_records(tf::buffer<Record> &records, std::size_t n_keys,
                        tf::buffer<Index> &offsets, tf::buffer<Record> &scratch,
                        const KeyOf &key_of, const Before &before) -> void {
  const std::size_t n = records.size();
  const std::size_t n_blocks =
      (n + tet_record_block_size - 1) / tet_record_block_size;
  tf::buffer<Index> counts;
  counts.allocate_and_initialize(n_blocks * n_keys, Index(0));
  tf::parallel_for_each(
      tf::make_sequence_range(n_blocks),
      [&records, &counts, &key_of, n, n_keys](std::size_t b) {
        auto count = counts.begin() + std::ptrdiff_t(b * n_keys);
        const std::size_t end = std::min(n, (b + 1) * tet_record_block_size);
        for (std::size_t r = b * tet_record_block_size; r < end; ++r)
          ++count[std::size_t(key_of(records[r]))];
      });
  offsets.allocate(n_keys + 1);
  offsets[0] = 0;
  for (std::size_t k = 0; k < n_keys; ++k) {
    Index running = offsets[k];
    for (std::size_t b = 0; b < n_blocks; ++b) {
      auto &count = counts[b * n_keys + k];
      const Index start = running;
      running += count;
      count = start;
    }
    offsets[k + 1] = running;
  }
  scratch.allocate(n);
  tf::parallel_for_each(
      tf::make_sequence_range(n_blocks),
      [&records, &counts, &scratch, &key_of, n, n_keys](std::size_t b) {
        auto cursor = counts.begin() + std::ptrdiff_t(b * n_keys);
        const std::size_t end = std::min(n, (b + 1) * tet_record_block_size);
        for (std::size_t r = b * tet_record_block_size; r < end; ++r)
          scratch[std::size_t(cursor[std::size_t(key_of(records[r]))]++)] =
              records[r];
      });
  tf::parallel_for_each(
      tf::make_sequence_range(n_keys),
      [&offsets, &scratch, &before](std::size_t k) {
        std::sort(scratch.begin() + std::ptrdiff_t(offsets[k]),
                  scratch.begin() + std::ptrdiff_t(offsets[k + 1]), before);
      },
      tf::checked);
  std::swap(records, scratch);
}

} // namespace tf::topology::cdt::dt3

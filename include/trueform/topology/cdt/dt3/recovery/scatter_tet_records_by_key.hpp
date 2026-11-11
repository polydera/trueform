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
#include "../../../../core/config/parallel_workers.hpp"
#include "../../../../core/config/verify.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "../tet_execution_tuning.hpp"
#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Group the records the sources state by key: `each(source, emit)` emits
/// the records of each source below `n_sources`, `key_of` names a record's
/// key below `n_keys`, block `k` of `offsets` is key `k`'s run in `out`, and
/// a run holds `value_of` of its records in the order they were emitted.
/// Blocked counts on coarse blocks of the key and one prefix scatter the
/// records into their coarse blocks' ranges of `staged`, and each coarse
/// block then places its own records by counts on its own keys; one worker,
/// or work below `tet_execution_tuning::record_serial_work` sources plus
/// keys, counts on the keys themselves, one lane and one scatter, as the
/// serial branch of @ref rank_tet_cells and the counting sorts do.
/// Either way the work is the records plus the keys, never their product.
/// Where a record is its own value the lane scatters into `staged` and
/// swaps it with `out`, so `each` may read what `out` holds; otherwise it
/// scatters into `out` directly, and `each` reads nothing `out` holds.
template <typename Index, typename Staged, typename Value, typename Each,
          typename KeyOf, typename ValueOf>
auto scatter_tet_records_by_key(std::size_t n_sources, std::size_t n_keys,
                                const Each &each, const KeyOf &key_of,
                                const ValueOf &value_of,
                                tf::buffer<Index> &offsets,
                                tf::buffer<Staged> &staged,
                                tf::buffer<Index> &counts,
                                tf::buffer<Value> &out) -> void {
  if constexpr (tf::core::verification)
    for (std::size_t source = 0; source < n_sources; ++source)
      each(source, [&key_of, n_keys](const Staged &record) {
        tf::core::verify(std::size_t(key_of(record)) < n_keys,
                         "a record names a key below the key count");
      });
  offsets.allocate(n_keys + 1);
  offsets[0] = 0;
  if (!n_keys) {
    staged.clear();
    out.clear();
    return;
  }
  if (n_sources + n_keys < tet_execution_tuning::record_serial_work ||
      tf::core::parallel_workers() < 2) {
    std::fill(offsets.begin() + 1, offsets.end(), Index(0));
    for (std::size_t source = 0; source < n_sources; ++source)
      each(source, [&key_of, &offsets](const Staged &record) {
        ++offsets[std::size_t(key_of(record)) + 1];
      });
    Index total = 0;
    for (std::size_t k = 1; k <= n_keys; ++k) {
      const Index held = offsets[k];
      offsets[k] = total;
      total += held;
    }
    const auto place = [&each, &key_of, &value_of, &offsets, n_sources,
                        total](tf::buffer<Value> &placed) {
      placed.allocate(std::size_t(total));
      for (std::size_t source = 0; source < n_sources; ++source)
        each(source,
             [&key_of, &value_of, &offsets, &placed](const Staged &record) {
               placed[std::size_t(offsets[std::size_t(key_of(record)) + 1]++)] =
                   value_of(record);
             });
    };
    if constexpr (std::is_same_v<Staged, Value>) {
      place(staged);
      std::swap(staged, out);
    } else
      place(out);
    return;
  }
  unsigned shift = 0;
  while ((n_keys - 1) >> shift >= tet_execution_tuning::record_coarse_keys)
    ++shift;
  const std::size_t width = ((n_keys - 1) >> shift) + 1;
  const std::size_t grain =
      std::max(tet_execution_tuning::scan_block_rows,
               tet_execution_tuning::sources_per_coarse_key * width);
  const std::size_t blocks = (n_sources + grain - 1) / grain;
  counts.allocate((blocks + 1) * width);
  tf::parallel_for_each(
      tf::make_sequence_range(blocks),
      [&each, &key_of, &counts, n_sources, width, grain,
       shift](std::size_t block) {
        const auto count = counts.data() + block * width;
        std::fill(count, count + width, Index(0));
        const auto last = std::min(n_sources, (block + 1) * grain);
        for (auto source = block * grain; source < last; ++source)
          each(source, [&key_of, count, shift](const Staged &record) {
            ++count[std::size_t(key_of(record)) >> shift];
          });
      },
      tf::checked(tet_execution_tuning::serial_scan_blocks));
  const auto cursor = counts.data() + blocks * width;
  std::fill(cursor, cursor + width, Index(0));
  for (std::size_t block = 0; block < blocks; ++block)
    for (std::size_t column = 0; column < width; ++column)
      cursor[column] += counts[block * width + column];
  Index total = 0;
  for (std::size_t column = 0; column < width; ++column) {
    offsets[column << shift] = total;
    const Index held = cursor[column];
    cursor[column] = total;
    total += held;
  }
  offsets[n_keys] = total;
  for (std::size_t block = 0; block < blocks; ++block)
    for (std::size_t column = 0; column < width; ++column) {
      auto &count = counts[block * width + column];
      const Index held = count;
      count = cursor[column];
      cursor[column] += held;
    }
  staged.allocate(std::size_t(total));
  tf::parallel_for_each(
      tf::make_sequence_range(blocks),
      [&each, &key_of, &counts, &staged, n_sources, width, grain,
       shift](std::size_t block) {
        const auto at = counts.data() + block * width;
        const auto last = std::min(n_sources, (block + 1) * grain);
        for (auto source = block * grain; source < last; ++source)
          each(source, [&key_of, &staged, at, shift](const Staged &record) {
            staged[std::size_t(at[std::size_t(key_of(record)) >> shift]++)] =
                record;
          });
      },
      tf::checked(tet_execution_tuning::serial_scan_blocks));
  out.allocate(std::size_t(total));
  struct lane_t {
    tf::buffer<Index> cursor;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(width),
      [&key_of, &value_of, &offsets, &staged, &out, n_keys,
       shift](std::size_t column, lane_t &lane) {
        const std::size_t first_key = column << shift;
        const std::size_t span =
            std::min(n_keys, (column + 1) << shift) - first_key;
        const auto first = std::size_t(offsets[first_key]);
        const auto last = std::size_t(offsets[first_key + span]);
        auto &cursor = lane.cursor;
        cursor.allocate(span);
        std::fill(cursor.begin(), cursor.end(), Index(0));
        for (auto at = first; at != last; ++at)
          ++cursor[std::size_t(key_of(staged[at])) - first_key];
        Index running = Index(first);
        for (std::size_t k = 0; k < span; ++k) {
          const Index held = cursor[k];
          cursor[k] = running;
          if (k)
            offsets[first_key + k] = running;
          running += held;
        }
        for (auto at = first; at != last; ++at) {
          const auto &record = staged[at];
          out[std::size_t(cursor[std::size_t(key_of(record)) - first_key]++)] =
              value_of(record);
        }
        if constexpr (tf::core::verification)
          tf::core::verify(cursor[span - 1] == Index(last),
                           "a coarse block places its records in its range");
      },
      lane_t{}, tf::checked(tet_execution_tuning::serial_rank_blocks));
}

} // namespace tf::topology::cdt::dt3

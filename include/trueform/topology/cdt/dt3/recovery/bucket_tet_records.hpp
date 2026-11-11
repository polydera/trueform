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
#include "./scatter_tet_records_by_key.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Order `records` by `key_of`, a key below `n_keys`, and each key's run by
/// `before`: grouped by key in their own order (@ref
/// scatter_tet_records_by_key, staged through `scratch` on `counts`), block
/// `k` of `offsets` key `k`'s run, and the runs sorted apart in parallel.
template <typename Record, typename Index, typename KeyOf, typename Before>
auto bucket_tet_records(tf::buffer<Record> &records, std::size_t n_keys,
                        tf::buffer<Index> &offsets, tf::buffer<Record> &scratch,
                        tf::buffer<Index> &counts, const KeyOf &key_of,
                        const Before &before) -> void {
  scatter_tet_records_by_key(
      records.size(), n_keys,
      [&records](std::size_t r, const auto &emit) { emit(records[r]); },
      key_of, [](const Record &record) -> const Record & { return record; },
      offsets, scratch, counts, records);
  tf::parallel_for_each(
      tf::make_sequence_range(n_keys),
      [&offsets, &records, &before](std::size_t k) {
        std::sort(records.begin() + std::ptrdiff_t(offsets[k]),
                  records.begin() + std::ptrdiff_t(offsets[k + 1]), before);
      },
      tf::checked);
}

} // namespace tf::topology::cdt::dt3

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
#include "./tet_site.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::topology::cdt::dt3 {

inline constexpr unsigned tet_weld_radix_bits = 11;

/// The order two weld records weld in: Morton key, then position, name and
/// input slot, so equal positions are adjacent with the lowest name first.
template <typename Index, typename Coord>
auto tet_weld_record_precedes(const tet_site<Index, Coord> &a,
                              const tet_site<Index, Coord> &b) -> bool {
  if (a.key != b.key)
    return a.key < b.key;
  return a.floor < b.floor ||
         (a.floor == b.floor &&
          (a.id < b.id || (a.id == b.id && a.input < b.input)));
}

/// Sort `records` into @ref tet_weld_record_precedes order serially: a
/// least-significant-digit radix pass per digit the largest key holds,
/// through `scratch` and `counts`, then each run of one key ordered on its
/// own; the runs are short, a key naming a cell of the halved lattice.
template <typename Index, typename Coord>
auto sort_tet_weld_records(tf::buffer<tet_site<Index, Coord>> &records,
                           tf::buffer<tet_site<Index, Coord>> &scratch,
                           tf::buffer<std::size_t> &counts) -> void {
  const std::size_t n = records.size();
  if (n < 2)
    return;
  const std::size_t radix_size = std::size_t(1) << tet_weld_radix_bits;
  const std::uint64_t radix_mask = std::uint64_t(radix_size - 1);
  std::uint64_t largest = 0;
  for (const auto &record : records)
    largest = std::max(largest, record.key);
  std::size_t pass_count = 0;
  do {
    ++pass_count;
    largest >>= tet_weld_radix_bits;
  } while (largest != 0);
  scratch.allocate(n);
  counts.allocate(radix_size);
  for (std::size_t pass = 0; pass < pass_count; ++pass) {
    std::fill(counts.begin(), counts.end(), std::size_t(0));
    const auto shift = unsigned(pass * tet_weld_radix_bits);
    for (const auto &record : records)
      ++counts[std::size_t((record.key >> shift) & radix_mask)];
    std::size_t next = 0;
    for (auto &count : counts) {
      const std::size_t after = next + count;
      count = next;
      next = after;
    }
    for (const auto &record : records)
      scratch[counts[std::size_t((record.key >> shift) & radix_mask)]++] =
          record;
    std::swap(records, scratch);
  }
  for (std::size_t run = 0; run < n;) {
    std::size_t end = run + 1;
    while (end < n && records[end].key == records[run].key)
      ++end;
    if (end - run > 1)
      std::sort(records.begin() + std::ptrdiff_t(run),
                records.begin() + std::ptrdiff_t(end),
                tet_weld_record_precedes<Index, Coord>);
    run = end;
  }
}

} // namespace tf::topology::cdt::dt3

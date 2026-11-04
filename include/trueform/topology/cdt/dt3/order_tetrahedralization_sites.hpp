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
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./tet_execution_tuning.hpp"
#include "./tet_site_level.hpp"
#include "./tet_site_position.hpp"
#include "./tet_site_priority.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// State the insertion order of the welded sites: levels descending
/// (@ref tet_site_level), the sites of one level in storage order, which is
/// their Morton order, by counts on the levels and one prefix.
template <bool Parallel = false, typename Owner>
auto order_tetrahedralization_sites(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  const std::size_t n = tet_site_count(owner);
  owner._order.allocate(n);
  const auto level_of = [&owner](std::size_t i) {
    return tet_site_level(
        tet_site_priority(std::uint64_t(tet_site_name(owner, Index(i)))));
  };
  if constexpr (!Parallel) {
    std::size_t starts[tet_site_levels + 1] = {};
    for (std::size_t i = 0; i < n; ++i)
      ++starts[tet_site_levels - 1 - level_of(i)];
    std::size_t next = 0;
    for (auto &start : starts) {
      const std::size_t held = start;
      start = next;
      next += held;
    }
    for (std::size_t i = 0; i < n; ++i)
      owner._order[starts[tet_site_levels - 1 - level_of(i)]++] = Index(i);
  } else {
    static constexpr std::size_t grain = tet_execution_tuning::scan_block_rows;
    const std::size_t blocks = (n + grain - 1) / grain;
    auto &counts = owner._block_counts;
    counts.allocate(blocks * tet_site_levels);
    tf::parallel_for_each(
        tf::make_sequence_range(blocks),
        [&counts, &level_of, n](std::size_t block) {
          auto count = counts.data() + block * tet_site_levels;
          std::fill(count, count + tet_site_levels, Index(0));
          const auto last = std::min(n, (block + 1) * grain);
          for (auto i = block * grain; i < last; ++i)
            ++count[tet_site_levels - 1 - level_of(i)];
        },
        tf::checked(tet_execution_tuning::serial_scan_blocks));
    Index next = 0;
    for (std::size_t level = 0; level < tet_site_levels; ++level)
      for (std::size_t block = 0; block < blocks; ++block) {
        auto &count = counts[block * tet_site_levels + level];
        const Index held = count;
        count = next;
        next += held;
      }
    tf::parallel_for_each(
        tf::make_sequence_range(blocks),
        [&owner, &counts, &level_of, n](std::size_t block) {
          auto cursor = counts.data() + block * tet_site_levels;
          const auto last = std::min(n, (block + 1) * grain);
          for (auto i = block * grain; i < last; ++i)
            owner._order[std::size_t(
                cursor[tet_site_levels - 1 - level_of(i)]++)] = Index(i);
        },
        tf::checked(tet_execution_tuning::serial_scan_blocks));
  }
}

} // namespace tf::topology::cdt::dt3

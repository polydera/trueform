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
#include "../../../core/algorithm/reduce.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/range.hpp"
#include "../../../core/views/mapped_range.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./advance_tet_name.hpp"
#include "./tet_execution_tuning.hpp"
#include <algorithm>
#include <cstddef>
#include <tbb/parallel_sort.h>

namespace tf::topology::cdt::dt3 {

/// Weld exact positions, retain the lowest canonical name and publish both
/// directions of the input-to-site map. Sites are ordered by coordinate, and
/// a name carried twice at one position keeps its lowest input slot.
template <bool Parallel = false, typename Owner>
auto weld_tetrahedralization_sites(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  auto &records = owner._site_scratch;
  const std::size_t n = records.size();
  const auto less = [](const auto &a, const auto &b) {
    return a.pt < b.pt ||
           (a.pt == b.pt &&
            (a.id < b.id || (a.id == b.id && a.input < b.input)));
  };
  const auto starts_site = [&records](std::size_t i) {
    return i == 0 || !(records[i].pt == records[i - 1].pt);
  };
  owner._index_map.f().allocate(n);

  if constexpr (!Parallel) {
    std::sort(records.begin(), records.end(), less);
    owner._sites.clear();
    owner._index_map.kept_ids().clear();
    for (std::size_t i = 0; i < n; ++i) {
      if (starts_site(i)) {
        owner._sites.push_back({records[i].id, records[i].pt});
        owner._index_map.kept_ids().push_back(records[i].input);
        owner._next_name = advance_tet_name(owner._next_name, records[i].id);
      }
      owner._index_map.f()[std::size_t(records[i].input)] =
          Index(owner._sites.size() - 1);
    }
  } else {
    tbb::parallel_sort(records.begin(), records.end(), less);
    const std::size_t grain = tet_execution_tuning::scan_block_rows;
    const auto blocks = (n + grain - 1) / grain;
    auto &offsets = owner._block_offsets;
    offsets.allocate(blocks + 1);
    offsets[0] = 0;
    tf::parallel_for_each(
        tf::make_sequence_range(blocks),
        [&](std::size_t block) {
          std::size_t heads = 0;
          const auto last = std::min(n, (block + 1) * grain);
          for (auto i = block * grain; i < last; ++i)
            heads += starts_site(i);
          offsets[block + 1] = heads;
        },
        tf::checked(tet_execution_tuning::serial_scan_blocks));
    for (std::size_t block = 0; block < blocks; ++block)
      offsets[block + 1] += offsets[block];
    owner._sites.allocate(offsets[blocks]);
    owner._index_map.kept_ids().allocate(offsets[blocks]);
    tf::parallel_for_each(
        tf::make_sequence_range(blocks),
        [&](std::size_t block) {
          auto site = Index(Index(offsets[block]) - 1);
          const auto last = std::min(n, (block + 1) * grain);
          for (auto i = block * grain; i < last; ++i) {
            if (starts_site(i)) {
              ++site;
              owner._sites[std::size_t(site)] = {records[i].id, records[i].pt};
              owner._index_map.kept_ids()[std::size_t(site)] =
                  records[i].input;
            }
            owner._index_map.f()[std::size_t(records[i].input)] = site;
          }
        },
        tf::checked(tet_execution_tuning::serial_scan_blocks));
    owner._next_name = tf::reduce(
        tf::make_mapped_range(
            tf::make_range(owner._sites),
            [](const auto &site) {
              return advance_tet_name(std::size_t(0), site.id);
            }),
        [](std::size_t a, std::size_t b) { return std::max(a, b); },
        owner._next_name, tf::checked);
  }
}

} // namespace tf::topology::cdt::dt3

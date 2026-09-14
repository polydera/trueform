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
#include "./pool_records.hpp"

#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::exact::door::pool {

/// Which cells can offer anything at all.
///
/// An offer is made by a member on behalf of a face, so a cell holding no
/// witnessed member will never ask its ladder a question, and the fact is
/// known before a single intercept exists.
///
/// The verdict is one disjoint store per cell, so the cell is the grain and
/// nothing is reduced. The census reads the verdicts back afterwards: a
/// silenced cell states the same refusals the offer pass would have stated,
/// one per member.
template <typename Int>
auto find_speaking_cells(const line_cells<Int> &cells,
                         const tf::buffer<int> &witness_offsets,
                         election_census &census, tf::buffer<char> &speaks)
    -> void {
  speaks.allocate(cells.key.size());
  tf::parallel_for_each(
      tf::make_sequence_range(cells.key.size()),
      [&cells, &witness_offsets, &speaks](std::size_t c) {
        bool held = false;
        for (int k = cells.offsets[c]; k < cells.offsets[c + 1] && !held; ++k) {
          const auto name = std::size_t(cells.member[std::size_t(k)]);
          held = witness_offsets[name] != witness_offsets[name + 1];
        }
        speaks[c] = char(held);
      },
      tf::checked);

#ifdef TF_POOL_CENSUS
  for (std::size_t c = 0; c < cells.key.size(); ++c) {
    if (speaks[c]) {
      ++census.speaking_cells;
      continue;
    }
    census.anchors_without_witness +=
        std::int64_t(cells.offsets[c + 1] - cells.offsets[c]);
  }
#else
  (void)census;
#endif
}

} // namespace tf::exact::door::pool

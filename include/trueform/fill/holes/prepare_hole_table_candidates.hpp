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
#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./hole_table.hpp"
#include "tbb/parallel_sort.h"
#include <array>
#include <cstddef>
#include <iterator>

namespace tf::fill {

/// Group the retained facets into the candidate block of each chord.
///
/// A row is `(chord low, chord high, interior apex, facet)`, which is its own
/// sort key: the chord manufactures the block and the apex orders the rows
/// inside it, so a stage that keeps the first minimum keeps the lowest apex.
template <typename Index>
auto prepare_hole_table_candidates(tf::fill::hole_table<Index> &table) -> void {
  const auto &facets = table.facets;
  auto &rows = table.candidates.data_buffer();
  rows.allocate(facets.size());
  tf::parallel_for_each(
      tf::make_sequence_range(facets.size()),
      [&facets, &rows](std::size_t f) {
        const auto facet = facets[f];
        rows[f] = {facet[0], facet[2], facet[1], Index(f)};
      },
      tf::checked);
  tbb::parallel_sort(rows.begin(), rows.end());

  auto &offsets = table.candidates.offsets_buffer();
  offsets.clear();
  tf::compute_offsets(rows, std::back_inserter(offsets), Index(0),
                      [](const std::array<Index, 4> &x,
                         const std::array<Index, 4> &y) {
                        return x[0] == y[0] && x[1] == y[1];
                      });
}

} // namespace tf::fill

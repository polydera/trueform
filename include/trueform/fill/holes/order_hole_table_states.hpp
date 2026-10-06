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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./hole_table.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::fill {

/// Give every state its candidate block and its band.
///
/// A query is `(chord low, chord high, state)`. Sorted, it walks the candidate
/// blocks once, so each state is handed the block of its own chord or `-1`
/// where no facet spans it; counting the queries by span then places each
/// state in the band its recurrence lets it be solved in.
template <typename Index>
auto order_hole_table_states(Index n, tf::fill::hole_table<Index> &table)
    -> void {
  const auto &facets = table.facets;
  auto &queries = table.queries;
  const std::size_t kept = facets.size();
  const std::size_t states = 2 * kept + 1;
  queries.allocate(states);
  tf::parallel_for_each(
      tf::make_sequence_range(kept),
      [&facets, &queries](std::size_t f) {
        const auto facet = facets[f];
        queries[2 * f] = {facet[0], facet[1], Index(2 * f)};
        queries[2 * f + 1] = {facet[1], facet[2], Index(2 * f + 1)};
      },
      tf::checked);
  queries[2 * kept] = {Index(0), Index(n - 1), Index(2 * kept)};
  tbb::parallel_sort(queries.begin(), queries.end());

  const auto &rows = table.candidates.data_buffer();
  const auto &starts = table.candidates.offsets_buffer();
  const std::size_t blocks = table.candidates.size();
  table.block.allocate(states);
  std::size_t block = 0;
  for (const auto &query : queries) {
    while (block < blocks &&
           (rows[std::size_t(starts[block])][0] < query[0] ||
            (rows[std::size_t(starts[block])][0] == query[0] &&
             rows[std::size_t(starts[block])][1] < query[1])))
      ++block;
    const bool spans = block < blocks &&
                       rows[std::size_t(starts[block])][0] == query[0] &&
                       rows[std::size_t(starts[block])][1] == query[1];
    table.block[std::size_t(query[2])] = spans ? Index(block) : Index(-1);
  }

  auto &offsets = table.bands.offsets_buffer();
  offsets.allocate(std::size_t(n) + 1);
  std::fill(offsets.begin(), offsets.end(), Index(0));
  for (const auto &query : queries)
    ++offsets[std::size_t(query[1] - query[0]) + 1];
  for (std::size_t span = 1; span <= std::size_t(n); ++span)
    offsets[span] += offsets[span - 1];
  table.cursor.allocate(std::size_t(n));
  std::copy(offsets.begin(), offsets.end() - 1, table.cursor.begin());
  table.bands.data_buffer().allocate(states);
  for (const auto &query : queries)
    table.bands.data_buffer()[std::size_t(
        table.cursor[std::size_t(query[1] - query[0])]++)] = query[2];
}

} // namespace tf::fill

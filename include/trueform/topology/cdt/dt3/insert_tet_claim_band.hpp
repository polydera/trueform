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
#include "../../../core/algorithm/parallel_contains.hpp"
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/config/verify.hpp"
#include "../../../core/grain.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./grow_tet_claim_pools.hpp"
#include "./measure_tet_claim_density.hpp"
#include "./prepare_tet_claim_band.hpp"
#include "./redistribute_tet_claim_rows.hpp"
#include "./run_tet_claim_stream.hpp"
#include "./tet_claim_rows_asked.hpp"
#include "./tet_claim_scratch.hpp"
#include "./tet_claim_stream.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include "./tet_execution_tuning.hpp"
#include "./tet_state.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Insert the sites `[begin, end)` of the order in rounds, a barrier before
/// each. A round runs the streams concurrently while the last one inserted a
/// site and many sites are left, and otherwise one stream after another,
/// where no claim can collide and only rows stop a stream. Each barrier
/// measures the complex's rows per site, moves free rows to the streams
/// holding fewer than their sites left ask for at that density, and grows
/// the pools by what no stream could give; a round run in order after a
/// barrier that moved and grew none ends the band, its sites left to the
/// serial tail.
template <typename Owner, typename Index>
auto insert_tet_claim_band(Owner &owner, tet_claim_workspace<Index> &work,
                           std::size_t begin, std::size_t end) -> void {
  using Tuning = tet_execution_tuning;
  prepare_tet_claim_band(owner, work, begin, end);
  ++owner._stats.parallel_bands;
  auto left = end - begin;
  const auto pool = [&] {
    const auto density = measure_tet_claim_density(
        owner, work, work.inserted + end - begin - left);
    const bool moved = redistribute_tet_claim_rows(work, density);
    const bool grown = grow_tet_claim_pools(owner, work, density);
    if constexpr (tf::core::verification)
      tf::core::verify(
          std::size_t(owner._corners.size()) == std::size_t(Owner::max_tets) ||
              std::all_of(work.streams.begin(), work.streams.end(),
                          [density](const tet_claim_stream<Index> &stream) {
                            return stream.free_count >=
                                   tet_claim_rows_asked(stream, density);
                          }),
          "a barrier leaves every stream its ask while the cell space lasts");
    return grown || moved;
  };
  pool();
  bool concurrent = true;
  for (;;) {
    if (concurrent)
      tf::parallel_for_each(
          tf::make_sequence_range(work.streams.size()),
          [&](std::size_t stream) {
            run_tet_claim_stream(owner, work, stream,
                                 work.scratch[stream].value);
          },
          tf::grain(1));
    else
      for (std::size_t stream = 0; stream < work.streams.size(); ++stream)
        run_tet_claim_stream(owner, work, stream, work.scratch[stream].value);
    const auto before = left;
    left = 0;
    for (const auto &stream : work.streams)
      left += stream.end - stream.begin;
    if (!left)
      break;
    const bool moved = pool();
    if (!concurrent && !moved)
      break;
    concurrent = left < before && left > Tuning::serial_round_sites;
    ++owner._stats.pool_restarts;
  }
  if constexpr (tf::core::verification)
    tf::core::verify(
        !tf::parallel_contains(
            tf::make_sequence_range(work.ownership.size()),
            [&owner, &work](std::size_t i) {
              return work.ownership[i].load(std::memory_order_relaxed) !=
                         tet_unowned &&
                     owner._states[i] != tet_state::dead;
            },
            tf::checked),
        "a band ends with every claim released but the pools' rows");
  for (const auto &stream : work.streams)
    for (auto i = stream.begin; i < stream.end; ++i)
      work.tail.push_back(owner._order[i]);
  const auto inserted = end - begin - left;
  work.inserted += inserted;
  owner._stats.parallel_insertions += inserted;
  owner._stats.insertions += inserted;
}

} // namespace tf::topology::cdt::dt3

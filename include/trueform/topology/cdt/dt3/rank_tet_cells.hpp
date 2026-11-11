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
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/config/parallel_workers.hpp"
#include "../../../core/config/verify.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./canonical_tet_key.hpp"
#include "./tet_cell_key.hpp"
#include "./tet_execution_tuning.hpp"
#include "./tet_site_position.hpp"
#include "./tet_state.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// State the canonical compaction and return the cells it keeps: every
/// finite cell its rank among the finite cells ordered by their ascending
/// corner slots, every hull cell its rank after them among the hull cells
/// ordered by their ascending finite corner slots, every dead cell `none`;
/// `_cell_keys` holds every kept cell's key in rank order, each carrying the
/// slot order the copy publishes it in. Distinct cells hold distinct corner
/// sets, so the order is total and no row's place follows the row it stood
/// in. A key's first slot is its cell's smallest corner, so counts on it
/// and one prefix place every key in its corner's run, the hull cells' runs
/// after the finite cells', and each run is ordered on its own.
/// Serially, and at one worker, the counts are the sites'; in parallel they
/// are blocked counts of coarse blocks of the smallest corner, and each
/// coarse block then places its keys by the same counts on its own corners,
/// in parallel. Both state the one total order, which a checked build
/// asserts on every rank.
template <bool Parallel, typename Owner>
auto rank_tet_cells(Owner &owner) -> std::size_t {
  using Index = typename Owner::index_type;
  const std::size_t n = owner._corners.size();
  const std::size_t sites = tet_site_count(owner);
  auto &keys = owner._cell_keys;
  auto &offsets = owner._key_offsets;
  auto &compaction = owner._compaction;
  compaction.allocate(n);
  const auto smallest = [&owner](std::size_t i) {
    const auto corners = owner._corners[i];
    return std::size_t(std::min(std::min(corners[0], corners[1]),
                                std::min(corners[2], corners[3])));
  };
  const auto smallest_finite = [&owner](std::size_t i) {
    const auto corners = owner._corners[i];
    return std::size_t(std::min(std::min(corners[0], corners[1]), corners[2]));
  };
  const auto rank_run = [&compaction, &keys](std::size_t first,
                                             std::size_t last) {
    const std::size_t size = last - first;
    if (size > tet_execution_tuning::rank_run_lanes) {
      std::sort(keys.begin() + first, keys.begin() + last,
                [](const tet_cell_key<Index> &a, const tet_cell_key<Index> &b) {
                  return a.corners < b.corners;
                });
      for (auto rank = first; rank < last; ++rank)
        compaction[std::size_t(keys[rank].row)] = Index(rank);
      return;
    }
    std::array<tet_cell_key<Index>, tet_execution_tuning::rank_run_lanes> run;
    std::array<Index, tet_execution_tuning::rank_run_lanes> second, third,
        fourth;
    for (std::size_t k = 0; k < size; ++k) {
      run[k] = keys[first + k];
      second[k] = run[k].corners[1];
      third[k] = run[k].corners[2];
      fourth[k] = run[k].corners[3];
    }
    for (std::size_t k = 0; k < size; ++k) {
      const Index b = second[k], c = third[k], d = fourth[k];
      std::size_t below = 0;
      for (std::size_t j = 0; j < size; ++j)
        below += unsigned(second[j] < b) |
                 (unsigned(second[j] == b) &
                  (unsigned(third[j] < c) |
                   (unsigned(third[j] == c) & unsigned(fourth[j] < d))));
      keys[first + below] = run[k];
      compaction[std::size_t(run[k].row)] = Index(first + below);
    }
  };
  const auto rank_serially = [&]() -> std::size_t {
    const std::size_t width = 2 * sites;
    offsets.allocate(width + 1);
    std::fill(offsets.begin(), offsets.end(), Index(0));
    for (std::size_t i = 0; i < n; ++i) {
      const auto state = owner._states[i];
      if (state == tet_state::live)
        ++offsets[smallest(i)];
      else if (state == tet_state::infinite)
        ++offsets[sites + smallest_finite(i)];
    }
    for (std::size_t column = 1; column < width; ++column)
      offsets[column] += offsets[column - 1];
    const std::size_t finite = std::size_t(offsets[sites - 1]);
    const std::size_t live = std::size_t(offsets[width - 1]);
    keys.allocate(live);
    for (std::size_t i = 0; i < n; ++i) {
      const auto state = owner._states[i];
      if (state == tet_state::live) {
        const auto key = canonical_tet_key(owner._corners[i], Index(i));
        keys[std::size_t(--offsets[std::size_t(key.corners[0])])] = key;
      } else if (state == tet_state::infinite) {
        const auto key = canonical_tet_hull_key(owner._corners[i], Index(i));
        keys[std::size_t(--offsets[sites + std::size_t(key.corners[0])])] = key;
      } else
        compaction[i] = Owner::none;
    }
    offsets[width] = Index(live);
    owner._n_finite = finite;
    for (std::size_t column = 0; column < width; ++column)
      rank_run(std::size_t(offsets[column]), std::size_t(offsets[column + 1]));
    return live;
  };
  const auto rank_in_parallel = [&]() -> std::size_t {
    unsigned shift = 0;
    while ((sites - 1) >> shift >= tet_execution_tuning::rank_coarse_blocks)
      ++shift;
    const std::size_t coarse = ((sites - 1) >> shift) + 1;
    const std::size_t width = 2 * coarse;
    const std::size_t grain =
        std::max(tet_execution_tuning::scan_block_rows,
                 tet_execution_tuning::rank_rows_per_coarse_block * coarse);
    const std::size_t blocks = (n + grain - 1) / grain;
    auto &counts = owner._block_counts;
    counts.allocate((blocks + 1) * width);
    tf::parallel_for_each(
        tf::make_sequence_range(blocks),
        [&owner, &counts, &smallest, &smallest_finite, n, coarse, width, grain,
         shift](std::size_t block) {
          const auto count = counts.data() + block * width;
          std::fill(count, count + width, Index(0));
          const auto last = std::min(n, (block + 1) * grain);
          for (auto i = block * grain; i < last; ++i) {
            const auto state = owner._states[i];
            if (state == tet_state::live)
              ++count[smallest(i) >> shift];
            else if (state == tet_state::infinite)
              ++count[coarse + (smallest_finite(i) >> shift)];
          }
        },
        tf::checked(tet_execution_tuning::serial_scan_blocks));
    const auto cursor = counts.data() + blocks * width;
    std::fill(cursor, cursor + width, Index(0));
    for (std::size_t block = 0; block < blocks; ++block)
      for (std::size_t column = 0; column < width; ++column)
        cursor[column] += counts[block * width + column];
    offsets.allocate(width + 1);
    offsets[0] = 0;
    for (std::size_t column = 0; column < width; ++column) {
      offsets[column + 1] = offsets[column] + cursor[column];
      cursor[column] = offsets[column];
    }
    for (std::size_t block = 0; block < blocks; ++block)
      for (std::size_t column = 0; column < width; ++column) {
        auto &count = counts[block * width + column];
        const Index held = count;
        count = cursor[column];
        cursor[column] += held;
      }
    const std::size_t finite = std::size_t(offsets[coarse]);
    const std::size_t live = std::size_t(offsets[width]);
    owner._n_finite = finite;
    keys.allocate(live);
    tf::parallel_for_each(
        tf::make_sequence_range(blocks),
        [&owner, &counts, &keys, &compaction, n, coarse, width, grain,
         shift](std::size_t block) {
          const auto cursor = counts.data() + block * width;
          const auto last = std::min(n, (block + 1) * grain);
          for (auto i = block * grain; i < last; ++i) {
            const auto state = owner._states[i];
            if (state == tet_state::live) {
              const auto key = canonical_tet_key(owner._corners[i], Index(i));
              keys[std::size_t(
                  cursor[std::size_t(key.corners[0]) >> shift]++)] = key;
            } else if (state == tet_state::infinite) {
              const auto key =
                  canonical_tet_hull_key(owner._corners[i], Index(i));
              keys[std::size_t(
                  cursor[coarse + (std::size_t(key.corners[0]) >> shift)]++)] =
                  key;
            } else
              compaction[i] = Owner::none;
          }
        },
        tf::checked(tet_execution_tuning::serial_scan_blocks));
    const std::size_t span = std::size_t(1) << shift;
    auto &lane = owner._bucket_starts;
    lane.allocate(width * 2 * (span + 1));
    tf::parallel_for_each(
        tf::make_sequence_range(width),
        [&keys, &offsets, &rank_run, &lane, coarse, shift,
         span](std::size_t bucket) {
          const auto first = std::size_t(offsets[bucket]);
          const auto last = std::size_t(offsets[bucket + 1]);
          if (first == last)
            return;
          const std::size_t site = (bucket < coarse ? bucket : bucket - coarse)
                                   << shift;
          const auto starts = lane.data() + bucket * 2 * (span + 1);
          const auto cursor = starts + span + 1;
          std::fill(starts, starts + span + 1, Index(0));
          for (auto at = first; at != last; ++at)
            ++starts[std::size_t(keys[at].corners[0]) - site + 1];
          for (std::size_t s = 1; s <= span; ++s)
            starts[s] += starts[s - 1];
          std::copy(starts, starts + span, cursor);
          // The run a key belongs in is its corner's, so the keys standing
          // elsewhere are swapped into it until every run is filled; a run
          // is then ordered on its own, so which of its keys arrived first
          // never shows.
          for (std::size_t s = 0; s < span; ++s)
            while (cursor[s] < starts[s + 1]) {
              const auto at = first + std::size_t(cursor[s]);
              const auto home = std::size_t(keys[at].corners[0]) - site;
              if (home == s) {
                ++cursor[s];
                continue;
              }
              std::swap(keys[at], keys[first + std::size_t(cursor[home]++)]);
            }
          for (std::size_t s = 0; s < span; ++s)
            rank_run(first + std::size_t(starts[s]),
                     first + std::size_t(starts[s + 1]));
        },
        tf::checked(tet_execution_tuning::serial_rank_blocks));
    if constexpr (tf::core::verification)
      tf::core::verify(
          !tf::parallel_contains(
              tf::make_sequence_range(live ? live - 1 : 0),
              [&keys, finite](std::size_t rank) {
                return rank + 1 != finite &&
                       !(keys[rank].corners < keys[rank + 1].corners);
              },
              tf::checked),
          "the finite cells' keys ascend in rank order, and the hull cells' "
          "after them");
    return live;
  };
  if constexpr (!Parallel)
    return rank_serially();
  else {
    const bool serially = tf::core::parallel_workers() < 2;
    const std::size_t live = serially ? rank_serially() : rank_in_parallel();
    if constexpr (tf::core::verification) {
      const tf::buffer<tet_cell_key<Index>> ranked = keys;
      const tf::buffer<Index> placed = compaction;
      const std::size_t finite = owner._n_finite;
      const std::size_t other = serially ? rank_in_parallel() : rank_serially();
      bool same = other == live && finite == owner._n_finite &&
                  placed.size() == compaction.size();
      for (std::size_t i = 0; same && i < live; ++i)
        same = ranked[i].corners == keys[i].corners &&
               ranked[i].row == keys[i].row &&
               ranked[i].slots == keys[i].slots &&
               ranked[i].exchanged == keys[i].exchanged;
      for (std::size_t i = 0; same && i < placed.size(); ++i)
        same = placed[i] == compaction[i];
      tf::core::verify(same, "the serial and the parallel rank state one "
                             "order");
    }
    return live;
  }
}

} // namespace tf::topology::cdt::dt3

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
#include "../../../core/range.hpp"
#include "../../../core/views/offset_block_range.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "../../../core/views/slice.hpp"
#include "./canonical_tet_slots.hpp"
#include "./tet_cell_key.hpp"
#include "./tet_execution_tuning.hpp"
#include "./tet_site_position.hpp"
#include "./tet_state.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Rewrite the compaction of every finite cell to its rank among the finite
/// cells ordered by their ascending corner slots, each key carrying the slot
/// order the copy publishes it in. Distinct cells hold distinct corner sets,
/// so the order is total. Serially, a key's first slot is its cell's
/// smallest corner, so counts on it and one prefix place every key in its
/// corner's block and each block is sorted on its own. In parallel the
/// keys are first scattered into coarse blocks of the smallest corner by
/// blocked counts and one prefix, and each coarse block then places its
/// keys by the same counts on its own corners, in parallel.
template <bool Parallel, typename Owner>
auto rank_tet_cells(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  const std::size_t n = owner._corners.size();
  auto &keys = owner._cell_keys;
  auto &offsets = owner._key_offsets;
  keys.allocate(owner._n_finite);
  const auto key_cell = [&owner, &keys](std::size_t i) {
    if (owner._states[i] != tet_state::live)
      return;
    std::array<Index, 4> key;
    const auto slots = canonical_tet_slots(owner._corners[i], key);
    keys[std::size_t(owner._compaction[i])] = {key, Index(i), slots};
  };
  const auto sort_block = [](auto block) {
    std::sort(block.begin(), block.end(),
              [](const tet_cell_key<Index> &a, const tet_cell_key<Index> &b) {
                return a.corners < b.corners;
              });
  };
  const auto rank_cell = [&owner, &keys](std::size_t rank) {
    owner._compaction[std::size_t(keys[rank].row)] = Index(rank);
  };
  if constexpr (Parallel)
    tf::parallel_for_each(tf::make_sequence_range(n), key_cell, tf::checked);
  else
    for (std::size_t i = 0; i < n; ++i)
      key_cell(i);
  const std::size_t sites = tet_site_count(owner);
  owner._cell_key_scratch.allocate(keys.size());
  if constexpr (!Parallel) {
    offsets.allocate(sites + 1);
    std::fill(offsets.begin(), offsets.end(), Index(0));
    for (const auto &key : keys)
      ++offsets[std::size_t(key.corners[0])];
    for (std::size_t site = 1; site < sites; ++site)
      offsets[site] += offsets[site - 1];
    for (const auto &key : keys)
      owner._cell_key_scratch[std::size_t(
          --offsets[std::size_t(key.corners[0])])] = key;
    offsets[sites] = Index(keys.size());
    std::swap(keys, owner._cell_key_scratch);
    for (auto block : tf::make_offset_block_range(offsets, keys))
      sort_block(block);
    for (std::size_t rank = 0; rank < owner._n_finite; ++rank)
      rank_cell(rank);
  } else {
    unsigned shift = 0;
    while ((sites - 1) >> shift >= tet_execution_tuning::rank_coarse_blocks)
      ++shift;
    const std::size_t coarse = ((sites - 1) >> shift) + 1;
    static constexpr std::size_t grain = tet_execution_tuning::scan_block_rows;
    const std::size_t blocks = (keys.size() + grain - 1) / grain;
    auto &counts = owner._block_counts;
    counts.allocate(blocks * coarse);
    tf::parallel_for_each(
        tf::make_sequence_range(blocks),
        [&keys, &counts, coarse, shift](std::size_t block) {
          auto count = tf::make_range(counts.data() + block * coarse,
                                      counts.data() + (block + 1) * coarse);
          std::fill(count.begin(), count.end(), Index(0));
          const auto last = std::min(keys.size(), (block + 1) * grain);
          for (auto i = block * grain; i < last; ++i)
            ++count[std::size_t(keys[i].corners[0]) >> shift];
        },
        tf::checked(tet_execution_tuning::serial_scan_blocks));
    offsets.allocate(coarse + 1);
    Index next = 0;
    for (std::size_t bucket = 0; bucket < coarse; ++bucket) {
      offsets[bucket] = next;
      for (std::size_t block = 0; block < blocks; ++block) {
        auto &count = counts[block * coarse + bucket];
        const Index held = count;
        count = next;
        next += held;
      }
    }
    offsets[coarse] = next;
    tf::parallel_for_each(
        tf::make_sequence_range(blocks),
        [&owner, &keys, &counts, coarse, shift](std::size_t block) {
          auto cursor = counts.data() + block * coarse;
          const auto last = std::min(keys.size(), (block + 1) * grain);
          for (auto i = block * grain; i < last; ++i)
            owner._cell_key_scratch[std::size_t(
                cursor[std::size_t(keys[i].corners[0]) >> shift]++)] = keys[i];
        },
        tf::checked(tet_execution_tuning::serial_scan_blocks));
    std::swap(keys, owner._cell_key_scratch);
    struct coarse_scratch {
      tf::buffer<Index> starts;
      tf::buffer<tet_cell_key<Index>> placed;
    };
    const Index span = Index(std::size_t(1) << shift);
    tf::parallel_for_each(
        tf::make_sequence_range(coarse),
        [&keys, &offsets, &sort_block, shift, span](std::size_t bucket,
                                                    coarse_scratch &scratch) {
          const auto block = tf::slice(keys, std::size_t(offsets[bucket]),
                                       std::size_t(offsets[bucket + 1]));
          if (!block.size())
            return;
          const Index first = Index(bucket << shift);
          scratch.starts.allocate(std::size_t(span) + 1);
          std::fill(scratch.starts.begin(), scratch.starts.end(), Index(0));
          for (const auto &key : block)
            ++scratch.starts[std::size_t(key.corners[0] - first) + 1];
          for (std::size_t site = 1; site <= std::size_t(span); ++site)
            scratch.starts[site] += scratch.starts[site - 1];
          scratch.placed.allocate(block.size());
          for (const auto &key : block)
            scratch.placed[std::size_t(
                scratch.starts[std::size_t(key.corners[0] - first)]++)] = key;
          for (std::size_t site = std::size_t(span); site > 0; --site)
            scratch.starts[site] = scratch.starts[site - 1];
          scratch.starts[0] = 0;
          for (auto run :
               tf::make_offset_block_range(scratch.starts, scratch.placed))
            sort_block(run);
          std::copy(scratch.placed.begin(), scratch.placed.end(),
                    block.begin());
        },
        coarse_scratch{}, tf::checked(8));
    tf::parallel_for_each(tf::make_sequence_range(owner._n_finite), rank_cell,
                          tf::checked);
  }
}

} // namespace tf::topology::cdt::dt3

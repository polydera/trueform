/*
 * Copyright (c) 2025 XLAB
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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/parallel_iota.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/sidedness.hpp"
#include "../../core/tagged_sidedness.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/zip.hpp"
#include "tbb/parallel_sort.h"
#include <cstddef>

namespace tf::topology::sidedness {

/// @ingroup topology_components
/// @brief Aggregate the per-edge sidedness vote stream into per-component
///        `(operand_tag, sidedness)` entries.
///
/// Pipeline:
///   1. Sort an index permutation by `(my_component, your_tag)`.
///   2. Count votes per component; prefix-sum into a per-component
///      vote-block offset buffer. View the sorted permutation as an
///      offset-block range keyed by component.
///   3. Count distinct tags per component (parallel) — that's the
///      number of `tagged_sidedness` entries each component emits.
///   4. Prefix-sum entry counts into `relations.offsets_buffer()` and
///      allocate the data buffer.
///   5. Per component (parallel), walk tag runs and write one
///      majority-resolved entry per run, using the precomputed
///      offsets as write slots. Tie-break: positive > negative >
///      boundary.
///
/// Empty components (no votes) produce empty per-component blocks at
/// their natural index — `offsets[c] == offsets[c + 1]`.
///
/// @tparam Index Integer type used for component ids and tag ids.
/// @param vote_components Per-vote `my_component`.
/// @param vote_entries Per-vote `{your_tag, sidedness}`.
/// @param n_components Total number of components.
/// @param relations Out: per-component block of unique
///        `tagged_sidedness` entries, one per operand the component
///        contacts at an intersection edge.
template <typename Index>
auto aggregate_votes(
    const tf::buffer<Index> &vote_components,
    const tf::buffer<tf::tagged_sidedness<Index>> &vote_entries,
    Index n_components,
    tf::offset_block_buffer<Index, tf::tagged_sidedness<Index>> &relations)
    -> void {
  auto V = vote_components.size();

  // (1) Sort index permutation by (component, tag).
  tf::buffer<Index> order;
  order.allocate(V);
  tf::parallel_iota(order, Index(0));
  tbb::parallel_sort(order.begin(), order.end(), [&](Index a, Index b) {
    auto ca = vote_components[a], cb = vote_components[b];
    if (ca != cb)
      return ca < cb;
    return vote_entries[a].tag < vote_entries[b].tag;
  });

  // (2) Per-component vote counts → vote-block offsets (prefix sum).
  tf::buffer<Index> vote_counts;
  vote_counts.allocate(n_components);
  tf::parallel_fill(vote_counts, Index(0));
  for (std::size_t i = 0; i < V; ++i)
    ++vote_counts[vote_components[i]];

  tf::buffer<Index> vote_offsets;
  vote_offsets.allocate(n_components + 1);
  {
    Index running = 0;
    for (Index c = 0; c < n_components; ++c) {
      vote_offsets[c] = running;
      running += vote_counts[c];
    }
    vote_offsets[n_components] = running;
  }

  // Per-component blocks of vote indices (already sorted by tag within
  // each block because of (1)).
  auto vote_blocks = tf::make_offset_block_range(vote_offsets, order);

  // (3) Count distinct tags per component.
  tf::buffer<Index> entry_counts;
  entry_counts.allocate(n_components);
  tf::parallel_for_each(
      tf::zip(vote_blocks, entry_counts),
      [&](auto pair) {
        auto &&[block, count] = pair;
        if (block.size() == 0) {
          count = 0;
          return;
        }
        Index n = 1;
        Index prev_tag = vote_entries[block.begin()[0]].tag;
        for (std::size_t i = 1; i < block.size(); ++i) {
          Index t = vote_entries[block.begin()[i]].tag;
          if (t != prev_tag) {
            ++n;
            prev_tag = t;
          }
        }
        count = n;
      },
      tf::checked);

  // (4) Prefix-sum entry counts into the output offsets buffer; size
  //     the data buffer to the total.
  relations.offsets_buffer().allocate(n_components + 1);
  Index total_entries = 0;
  for (Index c = 0; c < n_components; ++c) {
    relations.offsets_buffer()[c] = total_entries;
    total_entries += entry_counts[c];
  }
  relations.offsets_buffer()[n_components] = total_entries;
  relations.data_buffer().allocate(total_entries);

  // (5) Per-component (parallel), walk tag runs, write majority entry.
  tf::parallel_for_each(
      tf::enumerate(vote_blocks),
      [&](auto pair) {
        auto &&[c, block] = pair;
        if (block.size() == 0)
          return;
        auto write_idx = relations.offsets_buffer()[static_cast<Index>(c)];

        std::size_t i = 0;
        while (i < block.size()) {
          Index tag = vote_entries[block.begin()[i]].tag;
          std::size_t j = i + 1;
          while (j < block.size() && vote_entries[block.begin()[j]].tag == tag)
            ++j;

          Index cnt_neg = 0, cnt_pos = 0, cnt_bnd = 0;
          for (std::size_t k = i; k < j; ++k) {
            switch (vote_entries[block.begin()[k]].side) {
            case tf::sidedness::on_negative_side:
              ++cnt_neg;
              break;
            case tf::sidedness::on_positive_side:
              ++cnt_pos;
              break;
            case tf::sidedness::on_boundary:
              ++cnt_bnd;
              break;
            case tf::sidedness::none:
              // Selection-only sentinel; never produced by classify_wedge,
              // so never appears in the vote stream.
              break;
            }
          }

          tf::sidedness winner;
          if (cnt_pos >= cnt_neg && cnt_pos >= cnt_bnd)
            winner = tf::sidedness::on_positive_side;
          else if (cnt_neg >= cnt_bnd)
            winner = tf::sidedness::on_negative_side;
          else
            winner = tf::sidedness::on_boundary;

          relations.data_buffer()[write_idx++] = {tag, winner};
          i = j;
        }
      },
      tf::checked);
}

} // namespace tf::topology::sidedness

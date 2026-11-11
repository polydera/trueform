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
#include "../../../core/algorithm/parallel_fill.hpp"
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/blocked_buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/config/verify.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./assign_tet_compaction.hpp"
#include "./canonical_tet_key.hpp"
#include "./rank_tet_cells.hpp"
#include "./tet_facet_is_protected.hpp"
#include "./tet_state.hpp"
#include "./tetrahedralization_owner.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Compact finite cells before hull cells, remapping neighbors, the site
/// cells and the recorded rows, the records of one row merged into its
/// first in record order. A compaction keeps the relative order of
/// its cells and moves the hint with its row, a retired hint falling back
/// to cell zero; a canonical one publishes every cell in
/// @ref rank_tet_cells order, each in the slot order its key carries, and
/// states the hint at the first row, so the rows it publishes and the row a
/// walk starts from follow the cells alone, whatever rows they stood in. It
/// reads the cells in rank order, so every row is written in sequence. A
/// cell's corners are its key's, so the corner lane is read no more once
/// the keys stand: the canonical copy writes the neighbors over it, then
/// the corners from the keys over the neighbor lane, and the two lanes
/// trade places.
template <bool Parallel = false, bool Canonical = false, typename Owner>
auto compact_tetrahedralization(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  const std::size_t n = owner._corners.size();
  std::size_t live = 0;
  if constexpr (Canonical)
    live = rank_tet_cells<Parallel>(owner);
  else
    live = assign_tet_compaction<Parallel>(owner);
  const std::size_t n_finite = owner._n_finite;
  if constexpr (!Canonical) {
    owner._corner_scratch.allocate(live);
    owner._neighbor_scratch.allocate(live);
    owner._state_scratch.allocate(live);
  }
  const bool protects = owner._protected_slots.size() != 0;
  if (protects) {
    owner._protected_scratch.allocate(live);
    owner._owner_scratch.allocate(4 * live);
  }
  const auto copy_lanes =
      [&owner,
       protects](std::size_t i, std::size_t to,
                 const std::array<std::size_t, 4> &slots,
                 typename tf::blocked_buffer<Index, 4>::reference neighbors) {
        for (std::size_t slot = 0; slot < 4; ++slot)
          neighbors[slot] =
              owner._compaction[std::size_t(owner._neighbors[i][slots[slot]])];
        if (protects) {
          std::uint8_t byte = 0;
          for (std::size_t slot = 0; slot < 4; ++slot) {
            const bool held =
                tet_facet_is_protected(owner, Index(i), slots[slot]);
            if (held)
              byte = std::uint8_t(byte | (1u << slot));
            owner._owner_scratch[4 * to + slot] =
                held ? owner._owners[4 * i + slots[slot]] : Owner::none;
          }
          owner._protected_scratch[to] = byte;
        }
      };
  if constexpr (!Canonical) {
    const auto copy_cell = [&owner, &copy_lanes](std::size_t i) {
      const Index to = owner._compaction[i];
      if (to == Owner::none)
        return;
      auto corners = owner._corner_scratch[std::size_t(to)];
      for (std::size_t slot = 0; slot < 4; ++slot)
        corners[slot] = owner._corners[i][slot];
      owner._state_scratch[std::size_t(to)] = owner._states[i];
      copy_lanes(i, std::size_t(to), {0, 1, 2, 3},
                 owner._neighbor_scratch[std::size_t(to)]);
    };
    if constexpr (Parallel)
      tf::parallel_for_each(tf::make_sequence_range(n), copy_cell, tf::checked);
    else
      for (std::size_t i = 0; i < n; ++i)
        copy_cell(i);
    std::swap(owner._corners, owner._corner_scratch);
    std::swap(owner._neighbors, owner._neighbor_scratch);
    std::swap(owner._states, owner._state_scratch);
  }
  if constexpr (Canonical) {
    const auto copy_neighbors = [&owner, &copy_lanes](std::size_t rank) {
      const auto &key = owner._cell_keys[rank];
      copy_lanes(
          std::size_t(key.row), rank,
          {canonical_tet_slot(key.slots, 0), canonical_tet_slot(key.slots, 1),
           canonical_tet_slot(key.slots, 2), canonical_tet_slot(key.slots, 3)},
          owner._corners[rank]);
    };
    const auto copy_corners = [&owner](std::size_t rank) {
      const auto &key = owner._cell_keys[rank];
      auto corners = owner._neighbors[rank];
      corners[0] = key.corners[0];
      corners[1] = key.corners[1];
      corners[2] = key.corners[key.exchanged ? 3 : 2];
      corners[3] = key.corners[key.exchanged ? 2 : 3];
      owner._states[rank] = tet_state::live;
    };
    const auto copy_hull_corners = [&owner, n_finite](std::size_t j) {
      const auto &key = owner._cell_keys[n_finite + j];
      auto corners = owner._neighbors[n_finite + j];
      corners[0] = key.corners[0];
      corners[1] = key.corners[key.exchanged ? 2 : 1];
      corners[2] = key.corners[key.exchanged ? 1 : 2];
      corners[3] = key.corners[3];
      owner._states[n_finite + j] = tet_state::infinite;
    };
    if constexpr (Parallel) {
      tf::parallel_for_each(tf::make_sequence_range(live), copy_neighbors,
                            tf::checked);
      tf::parallel_for_each(tf::make_sequence_range(n_finite), copy_corners,
                            tf::checked);
      tf::parallel_for_each(tf::make_sequence_range(live - n_finite),
                            copy_hull_corners, tf::checked);
    } else {
      for (std::size_t rank = 0; rank < live; ++rank)
        copy_neighbors(rank);
      for (std::size_t rank = 0; rank < n_finite; ++rank)
        copy_corners(rank);
      for (std::size_t j = 0; j < live - n_finite; ++j)
        copy_hull_corners(j);
    }
    std::swap(owner._corners, owner._neighbors);
    owner._corners.reallocate(live);
    owner._neighbors.reallocate(live);
    owner._states.reallocate(live);
    if constexpr (tf::core::verification)
      tf::core::verify(!tf::parallel_contains(
                           tf::make_sequence_range(live),
                           [&owner, live](std::size_t row) {
                             for (const Index side : owner._neighbors[row])
                               if (side < 0 || std::size_t(side) >= live)
                                 return true;
                             return false;
                           },
                           tf::checked),
                       "every published neighbor names a published row");
  }
  if (protects) {
    std::swap(owner._protected_slots, owner._protected_scratch);
    std::swap(owner._owners, owner._owner_scratch);
  }
  owner._site_cells.remap(owner._compaction, Owner::none);
  if (owner._record_written) {
    auto &at = owner._written_at;
    at.allocate(live);
    tf::parallel_fill(at, Owner::none);
    std::size_t kept = 0;
    for (std::size_t i = 0; i < owner._written.size(); ++i) {
      const auto record = owner._written[i];
      const Index to = owner._compaction[std::size_t(record.row)];
      if (to == Owner::none)
        continue;
      auto &first = at[std::size_t(to)];
      if (first != Owner::none) {
        owner._written[std::size_t(first)].writes += record.writes;
        continue;
      }
      first = Index(kept);
      owner._written[kept++] = {to, record.writes};
    }
    owner._written.erase_till_end(owner._written.begin() +
                                  std::ptrdiff_t(kept));
    owner._written_floor = 0;
    if constexpr (tf::core::verification)
      for (const auto &record : owner._written)
        tf::core::verify(std::size_t(record.row) < live &&
                             owner._states[std::size_t(record.row)] !=
                                 tet_state::dead,
                         "a compaction leaves the record no dead row");
  }
  if constexpr (Canonical) {
    owner._hint = Index(0);
  } else {
    const Index hint = owner._compaction[std::size_t(owner._hint)];
    owner._hint = hint == Owner::none ? Index(0) : hint;
  }
  owner._n_dead = 0;
  owner._canonical = Canonical;
  ++owner._compaction_epoch;
}

} // namespace tf::topology::cdt::dt3

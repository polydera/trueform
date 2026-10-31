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
#include "../../core/algorithm/parallel_transform.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/reallocate.hpp"
#include "../../core/views/slice.hpp"
#include "./elect_identity_components.hpp"
#include "./find_near_positions.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

namespace tf::intersect::graph {

/// THE NEAR TABLE, DEFERRED: the position sorts the band join reads (@ref
/// collapse_plane_identities), each kept whole as one block, so the join
/// runs only when a reader asks (@ref join_near_identity_rows). A row is an
/// identity as its producer stated it — an original by its flat vertex, a
/// standing created identity, or a gate class by its index — at its point;
/// a block ascends by point. A pair whose two members both stand below the
/// block's floor is not the block's: a carve's block states its mints alone.
template <typename Index, typename Point> struct near_identity_rows {
  struct row_t {
    Point point;
    Index rank;
    Index id;
  };
  tf::buffer<row_t> rows;
  /// `[offsets[k], offsets[k + 1])` is block `k`; empty when none stands
  tf::buffer<Index> offsets;
  tf::buffer<Index> floors;

  auto clear() -> void {
    rows.clear();
    offsets.clear();
    floors.clear();
  }
  auto n_blocks() const -> std::size_t {
    return offsets.size() == 0 ? std::size_t(0) : offsets.size() - 1;
  }
  /// One sorted block, its rows taken whole when it is the first.
  auto append(tf::buffer<row_t> &&block, Index floor) -> void {
    if (block.size() == 0)
      return;
    if (offsets.size() == 0)
      offsets.push_back(Index(0));
    if (rows.size() == 0)
      rows = std::move(block);
    else
      tf::core::append(block, rows);
    offsets.push_back(Index(rows.size()));
    floors.push_back(floor);
  }
  /// Every created identity's row speaks `forward(id)`; an original's never
  /// moves.
  template <typename Forward> auto renumber(const Forward &forward) -> void {
    tf::parallel_for_each(
        rows,
        [&forward](row_t &row) {
          if (row.rank != Index(original_identity_rank))
            row.id = forward(row.id);
        },
        tf::checked);
  }
  /// Every block of `other`, after these.
  auto append(const near_identity_rows &other) -> void {
    if (other.n_blocks() == 0)
      return;
    if (offsets.size() == 0)
      offsets.push_back(Index(0));
    const auto base = Index(rows.size());
    tf::core::append(other.rows, rows);
    for (std::size_t k = 1; k < other.offsets.size(); ++k)
      offsets.push_back(base + other.offsets[k]);
    tf::core::append(other.floors, floors);
  }
};

/// The pairs of every block of `table` at `window` (@ref
/// find_near_positions), each member as its row states it in the flat
/// currency past `n_flat` — a gate class through `fresh(cls)` — appended to
/// `pairs` with `a < b`, in no order and possibly repeated.
template <typename Index, typename Point, typename Int, typename Fresh>
auto join_near_identity_rows(const near_identity_rows<Index, Point> &table,
                             Int window, Index n_flat, const Fresh &fresh,
                             tf::buffer<std::array<Index, 2>> &pairs) -> void {
  using row_t = typename near_identity_rows<Index, Point>::row_t;
  const auto flat_of = [&](const row_t &row) -> Index {
    if (row.rank == Index(original_identity_rank))
      return row.id;
    return n_flat +
           (row.rank == Index(fresh_identity_rank) ? fresh(row.id) : row.id);
  };
  for (std::size_t k = 0; k < table.n_blocks(); ++k) {
    const auto block =
        tf::slice(tf::make_range(table.rows), std::size_t(table.offsets[k]),
                  std::size_t(table.offsets[k + 1]));
    const auto floor = table.floors[k];
    tf::buffer<std::array<Index, 2>> at;
    find_near_positions<Index>(
        block, [](const row_t &row) -> const Point & { return row.point; },
        window, at);
    tf::buffer<std::array<Index, 2>> stated;
    stated.allocate(at.size());
    tf::parallel_transform(
        at, stated,
        [&](const std::array<Index, 2> &pair) {
          const auto a = flat_of(block[std::size_t(pair[0])]);
          const auto b = flat_of(block[std::size_t(pair[1])]);
          return a < b ? std::array<Index, 2>{a, b}
                       : std::array<Index, 2>{b, a};
        },
        tf::checked);
    stated.erase_till_end(std::remove_if(
        stated.begin(), stated.end(), [floor](const std::array<Index, 2> &p) {
          return p[0] == p[1] || p[1] < floor;
        }));
    tf::core::append(stated, pairs);
  }
}

} // namespace tf::intersect::graph

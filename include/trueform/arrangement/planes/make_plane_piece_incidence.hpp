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
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./make_plane_arrangement_cells.hpp"
#include "./plane_arrangement.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::arrangement {

/// The incidence between the arrangement's pieces and its cells, addressable
/// from either side.
///
/// A cell's boundary is pieces and nothing else, so this IS the adjacency a
/// component flood walks — the cross-plane join is the piece ticket, the same
/// currency everywhere. Every covering member states its own slot, duplicates
/// included: a cell's rows on one piece are exactly one per member covering it,
/// which is what makes the depth readable at the piece itself.
template <typename Index> struct plane_piece_incidence {
  /// Per piece: `triangle * 3 + slot` for every slot naming it, ascending —
  /// the fill walks the slot table in order, so one triangle's rows are
  /// adjacent and a repeated naming is the previous row's triangle.
  tf::offset_block_buffer<Index, Index> rows_of_piece;
  /// Per cell: the piece each of its naming slots belongs to.
  tf::offset_block_buffer<Index, Index> pieces_of_cell;
};

/// Both identity spaces are dense and bounded, so counts plus one prefix hand
/// every piece and every cell its exact disjoint run — no records to sort.
template <typename Index, typename Int>
auto make_plane_piece_incidence(
    const plane_arrangement<Index, Int> &arrangement,
    const plane_arrangement_cells<Index> &cells)
    -> plane_piece_incidence<Index> {
  const auto slots = arrangement.slot_parents();
  plane_piece_incidence<Index> out;
  auto &piece_offsets = out.rows_of_piece.offsets_buffer();
  auto &cell_offsets = out.pieces_of_cell.offsets_buffer();
  // a cell store the build did not fill yields the empty incidence —
  // every block empty, every consumer loudly empty, never a stale read
  if (cells.cell_of.size() != slots.size()) {
    piece_offsets.allocate(
        std::size_t(arrangement.final_piece_ticket_extent()) + 1);
    std::fill(piece_offsets.begin(), piece_offsets.end(), Index(0));
    cell_offsets.allocate(std::size_t(cells.n_cells) + 1);
    std::fill(cell_offsets.begin(), cell_offsets.end(), Index(0));
    return out;
  }
  piece_offsets.allocate(std::size_t(arrangement.final_piece_ticket_extent()) +
                         2);
  cell_offsets.allocate(std::size_t(cells.n_cells) + 2);
  std::fill(piece_offsets.begin(), piece_offsets.end(), Index(0));
  std::fill(cell_offsets.begin(), cell_offsets.end(), Index(0));

  // Counts land one entry high, so the prefix leaves every key its cursor in
  // the entry before its own and the fill advances it into place.
  std::size_t n_rows = 0;
  for (std::size_t t = 0; t < slots.size(); ++t)
    for (int s = 0; s < 3; ++s) {
      const auto piece = slots[t][std::size_t(s)];
      if (piece == Index(-1))
        continue;
      ++piece_offsets[std::size_t(piece) + 2];
      ++cell_offsets[std::size_t(cells.cell_of[t]) + 2];
      ++n_rows;
    }
  for (std::size_t piece = 2; piece < piece_offsets.size(); ++piece)
    piece_offsets[piece] += piece_offsets[piece - 1];
  for (std::size_t cell = 2; cell < cell_offsets.size(); ++cell)
    cell_offsets[cell] += cell_offsets[cell - 1];

  out.rows_of_piece.data_buffer().allocate(n_rows);
  out.pieces_of_cell.data_buffer().allocate(n_rows);
  for (std::size_t t = 0; t < slots.size(); ++t)
    for (int s = 0; s < 3; ++s) {
      const auto piece = slots[t][std::size_t(s)];
      if (piece == Index(-1))
        continue;
      out.rows_of_piece
          .data_buffer()[std::size_t(piece_offsets[std::size_t(piece) + 1]++)] =
          Index(t) * Index(3) + Index(s);
      out.pieces_of_cell.data_buffer()[std::size_t(
          cell_offsets[std::size_t(cells.cell_of[t]) + 1]++)] = piece;
    }
  piece_offsets.erase_till_end(piece_offsets.end() - 1);
  cell_offsets.erase_till_end(cell_offsets.end() - 1);
  // sorted per cell, so the flood can skip a piece's repeated namings
  tf::parallel_for_each(
      tf::make_sequence_range(std::size_t(cells.n_cells)),
      [&](std::size_t cell) {
        auto block = out.pieces_of_cell[cell];
        std::sort(block.begin(), block.end());
      },
      tf::checked);
  return out;
}

} // namespace tf::arrangement

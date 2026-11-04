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
#include "../../../../core/algorithm/parallel_for_each.hpp"
#include "../../../../core/buffer.hpp"
#include "../../../../core/offset_block_buffer.hpp"
#include "../../../../core/views/sequence_range.hpp"
#include "../tet_finite_cell.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The rows one block of @ref state_tet_site_cells counts and scatters.
inline constexpr std::size_t tet_site_cell_block_rows = std::size_t(1) << 16;

/// State the cells of every site of `dt`: block `s` of `site_cells` is the
/// finite live cells holding site `s`, ascending. Counts and one prefix over
/// the rows, taken in blocks of rows so every block scatters into its own
/// disjoint ranges.
template <typename DT, typename Index>
auto state_tet_site_cells(const DT &dt,
                          tf::offset_block_buffer<Index, Index> &site_cells)
    -> void {
  const auto cells = dt.tets();
  const std::size_t n_rows = dt.n_tets(), n_sites = dt.n_sites();
  const std::size_t n_blocks =
      (n_rows + tet_site_cell_block_rows - 1) / tet_site_cell_block_rows;
  tf::buffer<Index> counts;
  counts.allocate_and_initialize(n_blocks * n_sites, Index(0));
  tf::parallel_for_each(
      tf::make_sequence_range(n_blocks),
      [&dt, &cells, &counts, n_rows, n_sites](std::size_t b) {
        auto count = counts.begin() + std::ptrdiff_t(b * n_sites);
        const std::size_t end = std::min(n_rows, (b + 1) * tet_site_cell_block_rows);
        for (std::size_t row = b * tet_site_cell_block_rows; row < end; ++row) {
          if (!tet_finite_cell(dt, Index(row)))
            continue;
          for (auto corner : cells[row])
            ++count[std::size_t(corner)];
        }
      });
  auto &offsets = site_cells.offsets_buffer();
  offsets.allocate(n_sites + 1);
  offsets[0] = 0;
  for (std::size_t s = 0; s < n_sites; ++s) {
    Index running = offsets[s];
    for (std::size_t b = 0; b < n_blocks; ++b) {
      auto &count = counts[b * n_sites + s];
      const Index start = running;
      running += count;
      count = start;
    }
    offsets[s + 1] = running;
  }
  auto &data = site_cells.data_buffer();
  data.allocate(std::size_t(offsets[n_sites]));
  tf::parallel_for_each(
      tf::make_sequence_range(n_blocks),
      [&dt, &cells, &counts, &data, n_rows, n_sites](std::size_t b) {
        auto cursor = counts.begin() + std::ptrdiff_t(b * n_sites);
        const std::size_t end = std::min(n_rows, (b + 1) * tet_site_cell_block_rows);
        for (std::size_t row = b * tet_site_cell_block_rows; row < end; ++row) {
          if (!tet_finite_cell(dt, Index(row)))
            continue;
          for (auto corner : cells[row])
            data[std::size_t(cursor[std::size_t(corner)]++)] = Index(row);
        }
      });
}

} // namespace tf::topology::cdt::dt3

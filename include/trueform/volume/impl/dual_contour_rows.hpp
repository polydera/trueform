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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/grain.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./dual_contour_cells.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tf {
namespace volume_detail {

/// @brief Per cell row: its x trim, how many output vertices and dual quads it
/// owns, and how many of its cells carry surface.
///
/// Outside the four x-case rows' crossing spans every corner sign in the row is
/// constant, so no cell is active and no y or z edge crosses — unless the four
/// rows disagree at a trim boundary, which widens it. That is the flying-edges
/// argument, and it is what keeps the empty part of a volume free.
///
/// @param meta Eight entries per cell row; [0] receives the vertex count, [1]
///        the quad count, [4] and [5] the trim, and the rest are zeroed.
/// @param active_counts One entry per cell row, offset by one, so the caller's
///        prefix turns it into that row's block of active records.
template <typename Index, typename Id>
auto count_cell_rows(const dc_grid &g, const unsigned char *x_cases,
                     const Id *x_meta, std::size_t x_meta_stride,
                     const int *x_min, const int *x_max, Index *meta,
                     std::ptrdiff_t *active_counts) -> void {
  const int cx = g.cx, cy = g.cy, cz = g.cz;
  const auto cell_rows = static_cast<std::ptrdiff_t>(cy) * cz;
  tf::parallel_for_each(
      tf::make_sequence_range(cell_rows),
      [&](std::ptrdiff_t row) {
        const int z = int(row / cy), y = int(row % cy);
        const auto *x0 = &x_cases[g.x_case_row(y, z)];
        const auto *x1 = &x_cases[g.x_case_row(y + 1, z)];
        const auto *x2 = &x_cases[g.x_case_row(y, z + 1)];
        const auto *x3 = &x_cases[g.x_case_row(y + 1, z + 1)];
        const auto *m0 = &x_meta[g.row(y, z) * x_meta_stride];
        const auto *m1 = &x_meta[g.row(y + 1, z) * x_meta_stride];
        const auto *m2 = &x_meta[g.row(y, z + 1) * x_meta_stride];
        const auto *m3 = &x_meta[g.row(y + 1, z + 1) * x_meta_stride];
        int x_left = 0, x_right = cx;
        if ((m0[0] | m1[0] | m2[0] | m3[0]) == 0) {
          if (x0[0] == x1[0] && x1[0] == x2[0] && x2[0] == x3[0]) {
            auto *empty = &meta[row * 8];
            for (int i = 0; i < 8; ++i)
              empty[i] = 0;
            active_counts[row + 1] = 0;
            return;
          }
        } else {
          const auto r0 = g.row(y, z);
          const auto r1 = g.row(y + 1, z);
          const auto r2 = g.row(y, z + 1);
          const auto r3 = g.row(y + 1, z + 1);
          x_left = std::min(std::min(x_min[r0], x_min[r1]),
                            std::min(x_min[r2], x_min[r3]));
          x_right = std::max(std::max(x_max[r0], x_max[r1]),
                             std::max(x_max[r2], x_max[r3]));
          if (x_left > 0 && ((x0[x_left] & 1u) != (x1[x_left] & 1u) ||
                             (x1[x_left] & 1u) != (x2[x_left] & 1u) ||
                             (x2[x_left] & 1u) != (x3[x_left] & 1u)))
            x_left = 0;
          if (x_right < cx && ((x0[x_right] & 2u) != (x1[x_right] & 2u) ||
                               (x1[x_right] & 2u) != (x2[x_right] & 2u) ||
                               (x2[x_right] & 2u) != (x3[x_right] & 2u)))
            x_right = cx;
        }

        Index n_cells = 0, n_xq = 0, n_yq = 0, n_zq = 0, n_active = 0;
        for (int x = x_left; x < x_right; ++x) {
          auto composite = cell_case(x0 + x, x1 + x, x2 + x, x3 + x);
          n_cells += g.interior(x, y, z)
                         ? components().count[composite]
                         : cell_vertex_count(
                               composite, retained_edges(x, y, z, cx, cy, cz));
          n_active += composite != 0u && composite != 0xffu;
          // owned interior edges: x edge (x, y, z), y edge (x, y, z),
          // z edge (x, y, z) with the 4 surrounding cells in range
          auto cases = static_cast<unsigned char>(composite & 3u);
          if (y >= 1 && z >= 1 &&
              (cases == k_left_inside || cases == k_right_inside))
            ++n_xq;
          auto c0 = static_cast<unsigned char>(composite & 1u);
          auto c2 = static_cast<unsigned char>((composite >> 2u) & 1u);
          auto c4 = static_cast<unsigned char>((composite >> 4u) & 1u);
          if (x >= 1 && z >= 1 && c0 != c2)
            ++n_yq;
          if (x >= 1 && y >= 1 && c0 != c4)
            ++n_zq;
        }
        auto *row_out = &meta[row * 8];
        row_out[0] = n_cells;
        row_out[1] = n_xq + n_yq + n_zq;
        row_out[2] = 0;
        row_out[3] = 0;
        row_out[4] = x_left;
        row_out[5] = x_right;
        row_out[6] = 0;
        row_out[7] = 0;
        active_counts[row + 1] = n_active;
      },
      tf::grain(k_dc_row_grain));
}

/// @brief Fill the compact record of every active cell — its x, its case and
/// the first vertex it owns — before any neighbour asks for it.
template <typename Index>
auto fill_active_cells(const dc_grid &g, const unsigned char *x_cases,
                       const Index *meta, const std::ptrdiff_t *active_offsets,
                       int *active_x, Index *active_base,
                       unsigned char *active_case) -> void {
  const int cx = g.cx, cy = g.cy, cz = g.cz;
  const auto cell_rows = static_cast<std::ptrdiff_t>(cy) * cz;
  tf::parallel_for_each(
      tf::make_sequence_range(cell_rows),
      [&](std::ptrdiff_t row) {
        const int z = int(row / cy), y = int(row % cy);
        const auto *x0 = &x_cases[g.x_case_row(y, z)];
        const auto *x1 = &x_cases[g.x_case_row(y + 1, z)];
        const auto *x2 = &x_cases[g.x_case_row(y, z + 1)];
        const auto *x3 = &x_cases[g.x_case_row(y + 1, z + 1)];
        Index id = meta[row * 8];
        auto active = active_offsets[row];
        const int x_end = int(meta[row * 8 + 5]);
        for (int x = int(meta[row * 8 + 4]); x < x_end; ++x) {
          const auto composite = cell_case(x0 + x, x1 + x, x2 + x, x3 + x);
          if (composite == 0u || composite == 0xffu)
            continue;
          active_x[active] = x;
          active_base[active] = id;
          active_case[active] = composite;
          ++active;
          id += g.interior(x, y, z)
                    ? components().count[composite]
                    : cell_vertex_count(composite,
                                        retained_edges(x, y, z, cx, cy, cz));
        }
      },
      tf::grain(k_dc_row_grain));
}

} // namespace volume_detail
} // namespace tf

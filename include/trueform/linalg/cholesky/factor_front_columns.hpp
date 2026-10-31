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
#include "../../core/algorithm/parallel_for.hpp"
#include "../../core/checked.hpp"
#include "./factor_diagonal_tile.hpp"
#include "./solve_panel_rows.hpp"
#include "./subtract_tile_product.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tf::linalg {

/// How many columns one step of @ref tf::linalg::factor_front_columns
/// factors, and the side of the tiles its work is cut into. The panel width
/// is a constant because it fixes the order every entry takes its updates
/// in: one sum per panel, its products from zero. The tiles only cut the work
/// a step does in parallel.
inline constexpr std::size_t cholesky_panel_width = 64;
inline constexpr std::size_t cholesky_tile_size = 64;

/// Factor the first `k` columns of an `m` square lower front in place: its
/// leading block into `L11`, the rows below into `L21 = F21 L11^-T`, and the
/// trailing block into the update `F22 - L21 L21^T`.
///
/// Column `j < k` stands at `head + j m`, rows `0..m`; column `j >= k` at
/// `tail + (j - k)(m - k)`, rows `k..m`. The columns go `cholesky_panel_width`
/// at a time, right-looking: factor the panel's diagonal tile, solve the rows
/// below it, then subtract the panel's product from everything after it, the
/// tiles of each step in parallel. Returns false where a pivot is not
/// positive or not finite.
inline auto factor_front_columns(double *head, double *tail, std::size_t m,
                                 std::size_t k) -> bool {
  const std::size_t P = tf::linalg::cholesky_panel_width;
  const std::size_t T = tf::linalg::cholesky_tile_size;
  const std::size_t rest = m - k;
  for (std::size_t p0 = 0; p0 < k; p0 += P) {
    const std::size_t p1 = std::min(p0 + P, k);
    const std::size_t width = p1 - p0;
    double *panel = head + p0 * m;
    if (!tf::linalg::factor_diagonal_tile(panel + p0, m, width))
      return false;
    if (p1 == m)
      return true;
    // Tiles start every T rows after the panel and again at k, so none
    // straddles the head and the tail.
    const std::size_t in_head = (k - p1 + T - 1) / T;
    const std::size_t tiles = in_head + (rest + T - 1) / T;
    auto first_of = [&](std::size_t tile) {
      return tile < in_head ? p1 + tile * T : k + (tile - in_head) * T;
    };
    auto last_of = [&](std::size_t tile) {
      return tile + 1 == in_head ? k : std::min(first_of(tile) + T, m);
    };
    tf::parallel_for(
        std::size_t(0), tiles,
        [&](std::size_t first, std::size_t last) {
          for (std::size_t tile = first; tile < last; ++tile)
            tf::linalg::solve_panel_rows(panel + first_of(tile), panel + p0, m,
                                         last_of(tile) - first_of(tile), width);
        },
        tf::checked(2));
    // The lower tiles of the trailing block, numbered row by row.
    tf::parallel_for(
        std::size_t(0), tiles * (tiles + 1) / 2,
        [&](std::size_t first, std::size_t last) {
          std::size_t row =
              std::size_t((std::sqrt(8.0 * double(first) + 1.0) - 1.0) / 2.0);
          while (row * (row + 1) / 2 > first)
            --row;
          while ((row + 1) * (row + 2) / 2 <= first)
            ++row;
          std::size_t column = first - row * (row + 1) / 2;
          for (std::size_t at = first; at < last; ++at) {
            const std::size_t i0 = first_of(row);
            const std::size_t j0 = first_of(column);
            double *c =
                j0 < k ? head + j0 * m + i0 : tail + (j0 - k) * rest + (i0 - k);
            tf::linalg::subtract_tile_product(
                panel + i0, panel + j0, m, width, last_of(row) - i0,
                last_of(column) - j0, c, j0 < k ? m : rest, row == column);
            if (++column > row) {
              column = 0;
              ++row;
            }
          }
        },
        tf::checked(2));
  }
  return true;
}

} // namespace tf::linalg

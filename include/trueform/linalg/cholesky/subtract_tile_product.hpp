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
#include <algorithm>
#include <cstddef>

namespace tf::linalg {

/// The register block of @ref tf::linalg::subtract_tile_product: rows by
/// columns of the tile held in accumulators across the whole width.
inline constexpr std::size_t cholesky_block_rows = 8;
inline constexpr std::size_t cholesky_block_columns = 4;

/// `c(i, j) -= sum_t a(i, t) b(j, t)` over a `rows` by `columns` tile, where
/// column `t` of `a` and `b` stands `t ld` further on and `c`'s column `j`
/// stands at `j ldc`.
///
/// Every entry sums its products from zero, `t` ascending, and is subtracted
/// once, whatever block it falls in. A `diagonal` tile is `a`'s rows against
/// themselves and writes its lower triangle only.
inline auto subtract_tile_product(const double *a, const double *b,
                                  std::size_t ld, std::size_t width,
                                  std::size_t rows, std::size_t columns,
                                  double *c, std::size_t ldc, bool diagonal)
    -> void {
  constexpr std::size_t R = tf::linalg::cholesky_block_rows;
  constexpr std::size_t C = tf::linalg::cholesky_block_columns;
  for (std::size_t j0 = 0; j0 < columns; j0 += C) {
    const std::size_t nc = std::min(C, columns - j0);
    for (std::size_t i0 = diagonal ? j0 / R * R : 0; i0 < rows; i0 += R) {
      const std::size_t nr = std::min(R, rows - i0);
      double sum[C][R] = {};
      if (nr == R && nc == C) {
        for (std::size_t t = 0; t < width; ++t) {
          const double *at = a + t * ld + i0;
          const double *bt = b + t * ld + j0;
          for (std::size_t jj = 0; jj < C; ++jj)
            for (std::size_t ii = 0; ii < R; ++ii)
              sum[jj][ii] += at[ii] * bt[jj];
        }
      } else {
        for (std::size_t t = 0; t < width; ++t) {
          const double *at = a + t * ld + i0;
          const double *bt = b + t * ld + j0;
          for (std::size_t jj = 0; jj < nc; ++jj)
            for (std::size_t ii = 0; ii < nr; ++ii)
              sum[jj][ii] += at[ii] * bt[jj];
        }
      }
      for (std::size_t jj = 0; jj < nc; ++jj) {
        double *column = c + (j0 + jj) * ldc + i0;
        for (std::size_t ii = 0; ii < nr; ++ii)
          if (!diagonal || i0 + ii >= j0 + jj)
            column[ii] -= sum[jj][ii];
      }
    }
  }
}

} // namespace tf::linalg

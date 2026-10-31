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
#include "./subtract_tile_product.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::linalg {

/// `x = x l^-T` in place for `rows` rows of a panel `width` wide, where `l` is
/// the factored diagonal tile; columns of both stand `ld` apart. Each entry
/// takes the entries before it in its row, ascending, and then its pivot.
inline auto solve_panel_rows(double *x, const double *l, std::size_t ld,
                             std::size_t rows, std::size_t width) -> void {
  constexpr std::size_t R = tf::linalg::cholesky_block_rows;
  for (std::size_t i0 = 0; i0 < rows; i0 += R) {
    const std::size_t nr = std::min(R, rows - i0);
    for (std::size_t t = 0; t < width; ++t) {
      double *column = x + t * ld + i0;
      double row[R];
      for (std::size_t ii = 0; ii < nr; ++ii)
        row[ii] = column[ii];
      for (std::size_t u = 0; u < t; ++u) {
        const double *before = x + u * ld + i0;
        const double factor = l[u * ld + t];
        for (std::size_t ii = 0; ii < nr; ++ii)
          row[ii] -= before[ii] * factor;
      }
      const double pivot = l[t * ld + t];
      for (std::size_t ii = 0; ii < nr; ++ii)
        column[ii] = row[ii] / pivot;
    }
  }
}

} // namespace tf::linalg

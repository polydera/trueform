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
#include <cmath>
#include <cstddef>

namespace tf::linalg {

/// One step of a forward solve over `r` interleaved lanes: `column` holds a
/// factor column from its pivot on, `rows` rows, and `x` the lanes of those
/// rows; the pivot's lanes divide by it, and the rows below take its product.
///
/// Every product a triangular solve takes is fused with the sum it enters
/// through `std::fma`, so each lane takes one operation sequence at any lane
/// count and under any contraction the compiler is allowed.
inline auto eliminate_column(double *x, const double *column,
                             std::size_t rows, std::size_t r) -> void {
  if (r == 1) {
    const double solved = x[0] / column[0];
    x[0] = solved;
    for (std::size_t q = 1; q < rows; ++q)
      x[q] = std::fma(-column[q], solved, x[q]);
    return;
  }
  for (std::size_t c = 0; c < r; ++c)
    x[c] /= column[0];
  for (std::size_t q = 1; q < rows; ++q)
    for (std::size_t c = 0; c < r; ++c)
      x[q * r + c] = std::fma(-column[q], x[c], x[q * r + c]);
}

} // namespace tf::linalg

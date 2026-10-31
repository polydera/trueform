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

/// `t = l l^T` in place over the lower triangle of a `width` square tile whose
/// column `j` stands at `j ld`, column by column: each takes the columns
/// before it and then its pivot. Returns false where a pivot is not positive
/// or not finite.
inline auto factor_diagonal_tile(double *t, std::size_t ld, std::size_t width)
    -> bool {
  for (std::size_t j = 0; j < width; ++j) {
    double *column = t + j * ld;
    for (std::size_t u = 0; u < j; ++u) {
      const double *before = t + u * ld;
      const double factor = before[j];
      for (std::size_t i = j; i < width; ++i)
        column[i] -= before[i] * factor;
    }
    const double pivot = column[j];
    if (!(pivot > 0.0) || !std::isfinite(pivot))
      return false;
    const double root = std::sqrt(pivot);
    column[j] = root;
    for (std::size_t i = j + 1; i < width; ++i)
      column[i] /= root;
  }
  return true;
}

} // namespace tf::linalg

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

/// How many partial sums a column's dot keeps side by side.
inline constexpr std::size_t cholesky_dot_partials = 8;

/// `c^T y` for one column `c` of `rows` rows and one lane of `y`, whose rows
/// stand `stride` apart.
///
/// Row `q` adds its product to partial `q` modulo the count, fused as
/// @ref tf::linalg::eliminate_column states, and the partials fold by halves:
/// the sum's order is the row count's alone, so a lane takes the same bits at
/// every stride.
inline auto column_dot(const double *c, const double *y, std::size_t rows,
                       std::size_t stride) -> double {
  constexpr std::size_t P = tf::linalg::cholesky_dot_partials;
  double sums[P] = {};
  const std::size_t full = rows / P * P;
  for (std::size_t q0 = 0; q0 < full; q0 += P)
    for (std::size_t j = 0; j < P; ++j)
      sums[j] = std::fma(c[q0 + j], y[(q0 + j) * stride], sums[j]);
  for (std::size_t q = full; q < rows; ++q)
    sums[q - full] = std::fma(c[q], y[q * stride], sums[q - full]);
  for (std::size_t half = P / 2; half > 0; half /= 2)
    for (std::size_t j = 0; j < half; ++j)
      sums[j] += sums[j + half];
  return sums[0];
}

} // namespace tf::linalg

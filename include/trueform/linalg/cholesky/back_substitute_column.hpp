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
#include "./column_dot.hpp"
#include <cmath>
#include <cstddef>

namespace tf::linalg {

/// @ref tf::linalg::back_substitute_column for one lane, whose rows stand
/// `stride` apart.
inline auto back_substitute_lane(double *x, const double *column,
                                 std::size_t rows, std::size_t stride)
    -> void {
  double value = x[0];
  if (rows > 1) {
    value -= tf::linalg::column_dot(column + 2, x + 2 * stride, rows - 2,
                                    stride);
    value = std::fma(-column[1], x[stride], value);
  }
  x[0] = value / column[0];
}

/// One step of a backward solve over `r` interleaved lanes: `column` holds a
/// factor column from its pivot on, `rows` rows, and `x` the lanes of those
/// rows, every one below the pivot solved; the pivot's lanes take the column's
/// products with them and divide by the pivot.
///
/// The rows from the second below the pivot on go by
/// @ref tf::linalg::column_dot, and the row right below it after: that row
/// was solved last, and standing outside the dot's partial sums it waits on
/// nothing but its own product. Products fuse as
/// @ref tf::linalg::eliminate_column states.
inline auto back_substitute_column(double *x, const double *column,
                                   std::size_t rows, std::size_t r) -> void {
  if (r == 1)
    return tf::linalg::back_substitute_lane(x, column, rows, 1);
  for (std::size_t c = 0; c < r; ++c)
    tf::linalg::back_substitute_lane(x + c, column, rows, r);
}

} // namespace tf::linalg

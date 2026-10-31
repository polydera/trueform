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
#include "../../core/buffer.hpp"
#include "../cholesky_factor.hpp"
#include "../cholesky_pattern.hpp"
#include "./eliminate_column.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::linalg {

/// `lanes = L^-1 lanes` in place over `r` interleaved right-hand sides: entry
/// `(k, c)` stands at `k r + c`, so one pass over the factor serves every
/// lane.
///
/// Each supernode takes the lanes of all its rows into `gathered`, one run
/// in its rows' order, walks its block in storage order, one
/// @ref tf::linalg::eliminate_column per column, and puts the lanes back.
template <typename Index>
auto solve_lower(const tf::cholesky_pattern<Index> &pattern,
                 const tf::cholesky_factor &factor,
                 tf::buffer<double> &lanes, std::size_t r,
                 tf::buffer<double> &gathered) -> void {
  for (std::size_t s = 0; s < pattern.number_of_supernodes(); ++s) {
    const std::size_t first = std::size_t(pattern.supernode_offsets[s]);
    const std::size_t k = std::size_t(pattern.supernode_offsets[s + 1]) - first;
    const Index *rows =
        pattern.rows.begin() + std::size_t(pattern.row_offsets[s]);
    const std::size_t m =
        std::size_t(pattern.row_offsets[s + 1] - pattern.row_offsets[s]);
    const double *block =
        factor.values.begin() + std::size_t(pattern.value_offsets[s]);
    double *own = lanes.data() + first * r;
    gathered.allocate(m * r);
    double *x = gathered.begin();
    std::copy(own, own + k * r, x);
    if (r == 1)
      for (std::size_t q = k; q < m; ++q)
        x[q] = lanes[std::size_t(rows[q])];
    else
      for (std::size_t q = k; q < m; ++q)
        for (std::size_t c = 0; c < r; ++c)
          x[q * r + c] = lanes[std::size_t(rows[q]) * r + c];
    for (std::size_t t = 0; t < k; ++t)
      tf::linalg::eliminate_column(x + t * r, block + t * m + t, m - t, r);
    std::copy(x, x + k * r, own);
    if (r == 1)
      for (std::size_t q = k; q < m; ++q)
        lanes[std::size_t(rows[q])] = x[q];
    else
      for (std::size_t q = k; q < m; ++q)
        for (std::size_t c = 0; c < r; ++c)
          lanes[std::size_t(rows[q]) * r + c] = x[q * r + c];
  }
}

} // namespace tf::linalg

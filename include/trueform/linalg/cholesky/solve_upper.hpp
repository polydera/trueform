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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../cholesky_factor.hpp"
#include "../cholesky_pattern.hpp"
#include "./back_substitute_column.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::linalg {

/// The backward solve of the supernodes `schedule[from..to)`, last to first,
/// over `r` interleaved lanes: each takes the lanes of all its rows into
/// `gathered`, one run in its rows' order, takes one
/// @ref tf::linalg::back_substitute_column per column, last to first, and
/// puts its own lanes back. It reads only the solved lanes of the rows below
/// its columns and writes only its own.
template <typename Index>
auto solve_upper_run(const tf::cholesky_pattern<Index> &pattern,
                     const tf::cholesky_factor &factor,
                     tf::buffer<double> &lanes, std::size_t r,
                     std::size_t from, std::size_t to,
                     tf::buffer<double> &gathered) -> void {
  for (std::size_t slot = to; slot-- > from;) {
    const std::size_t s = std::size_t(pattern.schedule[slot]);
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
    for (std::size_t t = k; t-- > 0;)
      tf::linalg::back_substitute_column(x + t * r, block + t * m + t, m - t,
                                         r);
    std::copy(x, x + k * r, own);
  }
}

/// `lanes = L^-T lanes` in place over `r` interleaved right-hand sides: each
/// column, last to first, reads the lanes of its own rows, which `L^T` states
/// on the row the column names.
///
/// The frontier of the pattern's schedule solves first through
/// @ref tf::linalg::solve_upper_run on `gathered`, and then its subtree blocks
/// in parallel, each on its own: a block reads the frontier's solved lanes
/// and writes only its own.
template <typename Index>
auto solve_upper(const tf::cholesky_pattern<Index> &pattern,
                 const tf::cholesky_factor &factor,
                 tf::buffer<double> &lanes, std::size_t r,
                 tf::buffer<double> &gathered) -> void {
  tf::linalg::solve_upper_run(pattern, factor, lanes, r,
                              pattern.frontier_first(),
                              pattern.number_of_supernodes(), gathered);
  tf::parallel_for_each(
      tf::make_sequence_range(pattern.number_of_blocks()),
      [&](std::size_t block, tf::buffer<double> &local) {
        tf::linalg::solve_upper_run(
            pattern, factor, lanes, r, std::size_t(pattern.block_offsets[block]),
            std::size_t(pattern.block_offsets[block + 1]), local);
      },
      tf::buffer<double>{});
}

} // namespace tf::linalg

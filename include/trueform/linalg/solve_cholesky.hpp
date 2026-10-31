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
#include "../core/algorithm/parallel_for.hpp"
#include "../core/buffer.hpp"
#include "../core/checked.hpp"
#include "./cholesky/solve_lower.hpp"
#include "./cholesky/solve_upper.hpp"
#include "./cholesky_factor.hpp"
#include "./cholesky_pattern.hpp"
#include "./cholesky_scratch.hpp"
#include <cstddef>

namespace tf {

/// @ingroup linalg
/// @brief Solve `A x = b` in place for `r` right-hand sides at once, over a
/// factor @ref tf::factor_cholesky accepted.
///
/// `rhs` holds `n r` values, entry `(i, c)` at `i r + c`, in the input's own
/// row order; it receives `x` the same way.
///
/// A lane's answer is the same bits at every `r`, every partition of the
/// pattern's schedule, every thread count and every run: each lane takes one
/// operation sequence whatever the lanes beside it, and each column's
/// arithmetic is the same whichever order the supernodes are taken in.
template <typename Index>
auto solve_cholesky(const tf::cholesky_pattern<Index> &pattern,
                    const tf::cholesky_factor &factor, tf::buffer<double> &rhs,
                    std::size_t r, tf::cholesky_scratch<Index> &scratch)
    -> void {
  const std::size_t n = pattern.size();
  auto &lanes = scratch.permuted;
  lanes.allocate(n * r);
  tf::parallel_for(
      std::size_t(0), n,
      [&](std::size_t first, std::size_t last) {
        for (std::size_t k = first; k < last; ++k)
          for (std::size_t c = 0; c < r; ++c)
            lanes[k * r + c] = rhs[std::size_t(pattern.order[k]) * r + c];
      },
      tf::checked);
  tf::linalg::solve_lower(pattern, factor, lanes, r, scratch.gathered);
  tf::linalg::solve_upper(pattern, factor, lanes, r, scratch.gathered);
  tf::parallel_for(
      std::size_t(0), n,
      [&](std::size_t first, std::size_t last) {
        for (std::size_t k = first; k < last; ++k)
          for (std::size_t c = 0; c < r; ++c)
            rhs[std::size_t(pattern.order[k]) * r + c] = lanes[k * r + c];
      },
      tf::checked);
}

} // namespace tf

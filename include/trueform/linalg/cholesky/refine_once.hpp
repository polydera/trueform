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
#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/algorithm/parallel_for.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../cholesky_factor.hpp"
#include "../cholesky_pattern.hpp"
#include "../cholesky_scratch.hpp"
#include "../solve_cholesky.hpp"
#include "../symmetric_lower.hpp"
#include "./symmetric_lower_residual.hpp"
#include <cmath>
#include <cstddef>

namespace tf::linalg {

/// The componentwise backward error an answer is accepted at: every row's
/// true residual within this much of `|b| + |A| |x|` on that row.
inline constexpr double cholesky_residual_tolerance = 1e-10;

/// Refine `x`, a solve of `A x = b` over `r` interleaved lanes, by one step
/// against its true residual, and state whether the refined answer holds.
///
/// The step solves the residual over the same factor and adds it, which
/// moves the answer toward the exact one. The refined answer holds where it
/// is finite and every row's recomputed residual is within
/// @ref tf::linalg::cholesky_residual_tolerance of `|b| + |A| |x|` there. The
/// measure is the row's own rather than `|b|`'s: a definite system of large
/// condition answers a backward-stable `x` whose residual is still large
/// beside `b`.
template <typename Index>
auto refine_once(const tf::symmetric_lower<Index> &matrix,
                 const tf::cholesky_pattern<Index> &pattern,
                 const tf::cholesky_factor &factor, const tf::buffer<double> &b,
                 std::size_t r, tf::cholesky_scratch<Index> &scratch,
                 tf::buffer<double> &x) -> bool {
  const std::size_t n = matrix.size();
  auto &residual = scratch.residual;
  auto &scale = scratch.scale;
  tf::linalg::symmetric_lower_residual(matrix, pattern, b, x, r, residual,
                                       scale);
  tf::solve_cholesky(pattern, factor, residual, r, scratch);
  tf::parallel_for(
      std::size_t(0), n * r,
      [&](std::size_t first, std::size_t last) {
        for (std::size_t k = first; k < last; ++k)
          x[k] += residual[k];
      },
      tf::checked);
  tf::linalg::symmetric_lower_residual(matrix, pattern, b, x, r, residual,
                                       scale);
  return !tf::parallel_contains(
      tf::make_sequence_range(n * r),
      [&](std::size_t k) {
        return !std::isfinite(x[k]) ||
               !(std::abs(residual[k]) <=
                 tf::linalg::cholesky_residual_tolerance * scale[k]);
      },
      tf::checked);
}

} // namespace tf::linalg

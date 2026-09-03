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

#include "../core/point.hpp"
#include "./meta.hpp"
#include <cstddef>

namespace tf::exact {

/// @ingroup exact
/// @brief Exact integer slab test with segment and box carried in one
///        common positive multiple of the `Int` lattice.
///
/// Every comparison is between two of the same rationals, so scaling
/// segment and box together scales numerators and denominators alike and
/// the verdict is the unscaled one's.
template <typename Int, typename Coord, std::size_t Dims>
auto segment_hits_aabb_scaled(const tf::point<Coord, Dims> &a,
                              const tf::point<Coord, Dims> &b,
                              const tf::point<Coord, Dims> &lo,
                              const tf::point<Coord, Dims> &hi) -> bool {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;

  T1 num_lo[Dims], num_hi[Dims];
  T1 den[Dims];
  for (std::size_t i = 0; i < Dims; ++i) {
    num_lo[i] = T1(lo[i]) - T1(a[i]);
    num_hi[i] = T1(hi[i]) - T1(a[i]);
    den[i] = T1(b[i]) - T1(a[i]);
    if (den[i] == T1(0)) {
      // Parallel to slab: hit iff a[i] is inside. Mark axis non-constraining.
      if (T1(a[i]) < T1(lo[i]) || T1(a[i]) > T1(hi[i]))
        return false;
      num_lo[i] = T1(0);
      num_hi[i] = T1(1);
      den[i] = T1(1);
    }
  }

  // (ni/di) <= (nj/dj), with one inequality flip per opposite-sign den.
  auto le_ratio = [](T1 ni, T1 di, T1 nj, T1 dj) -> bool {
    const bool si = di < T1(0);
    const bool sj = dj < T1(0);
    const T2 lhs = T2(ni) * T2(dj);
    const T2 rhs = T2(nj) * T2(di);
    return (si == sj) ? (lhs <= rhs) : (lhs >= rhs);
  };

  // den > 0: enter = num_lo/den, exit = num_hi/den; flipped if den < 0.
  auto enter_num = [&](std::size_t i) -> T1 {
    return den[i] >= T1(0) ? num_lo[i] : num_hi[i];
  };
  auto exit_num = [&](std::size_t i) -> T1 {
    return den[i] >= T1(0) ? num_hi[i] : num_lo[i];
  };

  // Per-axis interval must overlap [0, 1].
  for (std::size_t i = 0; i < Dims; ++i) {
    if (!le_ratio(enter_num(i), den[i], T1(1), T1(1)))
      return false;
    if (!le_ratio(T1(0), T1(1), exit_num(i), den[i]))
      return false;
  }

  // All pairs of intervals must overlap.
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < Dims; ++j)
      if (!le_ratio(enter_num(i), den[i], exit_num(j), den[j]))
        return false;

  return true;
}

/// @ingroup exact
/// @brief Exact integer slab test: closed-segment vs closed-AABB.
///
/// Per-axis entry/exit parameters `t = (slab - a[i]) / d[i]` are kept
/// as signed-denominator rationals; cross-axis comparisons reduce to
/// T2-width cross-multiplications (no division).
template <typename Int, std::size_t Dims>
auto segment_hits_aabb(const tf::point<Int, Dims> &a,
                       const tf::point<Int, Dims> &b,
                       const tf::point<Int, Dims> &lo,
                       const tf::point<Int, Dims> &hi) -> bool {
  return segment_hits_aabb_scaled<Int>(a, b, lo, hi);
}

} // namespace tf::exact

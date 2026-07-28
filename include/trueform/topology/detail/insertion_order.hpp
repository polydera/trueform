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
#include "./size_adaptive.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <cstdint>

namespace tf::topology::detail {

// Hilbert-curve index of a point on a 2^k x 2^k grid (k = k_order_bits).
// Hilbert order has far better locality than Morton (Z-order) — consecutive
// points stay spatially adjacent even for curve-like inputs, which keeps the
// per-insertion flip cascade local (O(1) amortized instead of super-linear).
inline constexpr std::uint32_t k_order_bits = 21;
inline auto hilbert_code(std::uint32_t x, std::uint32_t y) -> std::uint64_t {
  std::uint64_t d = 0;
  for (std::uint32_t s = k_order_bits; s-- > 0;) {
    std::uint32_t rx = (x >> s) & 1u;
    std::uint32_t ry = (y >> s) & 1u;
    d += static_cast<std::uint64_t>((3u * rx) ^ ry) << (2u * s);
    // Rotate the quadrant so the curve stays continuous.
    if (ry == 0u) {
      if (rx == 1u) {
        std::uint32_t m = (1u << k_order_bits) - 1u;
        x = m - x;
        y = m - y;
      }
      std::uint32_t t = x;
      x = y;
      y = t;
    }
  }
  return d;
}

template <typename Index>
auto brio_sort(Index *ord, const std::uint64_t *key, Index begin, Index lo,
               Index hi) -> void {
  auto cmp = [&](Index a, Index b) {
    // total order: tie-break equal Hilbert keys by index, so parallel
    // sort is deterministic (timing-independent) on duplicate cells
    auto ka = key[a - begin], kb = key[b - begin];
    return ka != kb ? ka < kb : a < b;
  };
  Index size = hi - lo;
  Index mid = lo;
  if (size >= 16) {
    mid = lo + (size * Index(3)) / Index(4);
    brio_sort(ord, key, begin, lo, mid);
  }
  // Hilbert-sort this round [mid, hi). Rounds are disjoint, so each sort is
  // independent; the largest rounds dominate and go parallel.
  if (hi - mid > Index(4096))
    tbb::parallel_sort(ord + mid, ord + hi, cmp);
  else
    std::sort(ord + mid, ord + hi, cmp);
}

/// The order points are inserted in: a biased randomized insertion order over
/// their Hilbert codes.
///
/// Locality is what keeps the walk that locates each point, and the flip
/// cascade that follows it, from degrading — consecutive points stay
/// spatially adjacent, so both start next to where the last one finished.
template <typename Points, typename Index>
auto build_insertion_order(const Points &points, Index begin, Index end,
                           tf::buffer<Index> &order,
                           tf::buffer<std::uint64_t> &keys) -> void {
  auto min_x = points[begin][0];
  auto min_y = points[begin][1];
  auto max_x = min_x;
  auto max_y = min_y;
  for (Index i = begin; i < end; ++i) {
    auto p = points[i];
    min_x = p[0] < min_x ? p[0] : min_x;
    min_y = p[1] < min_y ? p[1] : min_y;
    max_x = p[0] > max_x ? p[0] : max_x;
    max_y = p[1] > max_y ? p[1] : max_y;
  }
  // Reduce each axis to 21 bits so the interleave fits in 64 bits. Widen to
  // 64-bit before subtracting: converter-range int coords span the full
  // int32 range, so an int32 difference would overflow.
  auto span_x = static_cast<std::uint64_t>(static_cast<std::int64_t>(max_x) -
                                           static_cast<std::int64_t>(min_x));
  auto span_y = static_cast<std::uint64_t>(static_cast<std::int64_t>(max_y) -
                                           static_cast<std::int64_t>(min_y));
  auto span = span_x > span_y ? span_x : span_y;
  std::uint32_t shift = 0;
  while ((span >> shift) > 0x1fffffULL)
    ++shift;

  Index n = end - begin;
  order.allocate(static_cast<std::size_t>(n));
  // tiny inputs: any insertion order is O(1)-local for the remembering
  // walk; the shuffle + Hilbert coding would cost more than they save
  if (n < Index(64)) {
    for (Index k = 0; k < n; ++k)
      order[static_cast<std::size_t>(k)] = begin + k;
    return;
  }
    keys.clear();
  keys.allocate(static_cast<std::size_t>(n));
  auto fill_keys = [&](Index k_lo, Index k_hi) {
    for (Index k = k_lo; k != k_hi; ++k) {
      auto p = points[begin + k];
      auto x = static_cast<std::uint32_t>(
          static_cast<std::uint64_t>(static_cast<std::int64_t>(p[0]) -
                                     static_cast<std::int64_t>(min_x)) >>
          shift);
      auto y = static_cast<std::uint32_t>(
          static_cast<std::uint64_t>(static_cast<std::int64_t>(p[1]) -
                                     static_cast<std::int64_t>(min_y)) >>
          shift);
      order[static_cast<std::size_t>(k)] = begin + k;
      keys[static_cast<std::size_t>(k)] = hilbert_code(x, y);
    }
  };
  if (std::size_t(n) < topology::detail::k_serial_cutoff)
    fill_keys(Index(0), n);
  else
    tbb::parallel_for(tbb::blocked_range<Index>(Index(0), n),
                      [&](const tbb::blocked_range<Index> &r) {
                        fill_keys(r.begin(), r.end());
                      });

  // Biased Randomized Insertion Order (BRIO), matching CGAL's spatial_sort:
  // a random shuffle, then a multiscale recursion that Hilbert-sorts rounds
  // of geometrically increasing size (ratio 0.25). This inserts points in
  // randomized rounds of increasing density — each round well spread over the
  // domain — which makes incremental Delaunay O(N log N). Pure Hilbert order
  // alone is super-linear on structured inputs like a convex curve.
  Index *ord = &order[0];
  const std::uint64_t *key = &keys[0];

  std::uint64_t s =
      0x9e3779b97f4a7c15ULL ^ (static_cast<std::uint64_t>(n) + 1u) *
                                  0xff51afd7ed558ccdULL;
  auto next_rand = [&]() {
    s ^= s >> 33;
    s *= 0xff51afd7ed558ccdULL;
    s ^= s >> 33;
    return s;
  };
  for (Index k = n - 1; k > 0; --k) {
    Index j = static_cast<Index>(next_rand() % static_cast<std::uint64_t>(k + 1));
    std::swap(ord[k], ord[j]);
  }

  brio_sort(ord, key, begin, Index(0), n);
}

} // namespace tf::topology::detail

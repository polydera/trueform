/*
 * Copyright (c) 2025 XLAB
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

#include "trueform/core/buffer.hpp"
#include "trueform/core/range.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tf {
namespace ts {

// ============================================================================
// Bin index helpers (NumPy-compatible semantics).
// ============================================================================

/// @brief Bin index for equal-width slicing over `[lo, hi]`.
/// Returns -1 for NaN or out-of-range values. `v == hi` lands in the last
/// bin (inclusive upper edge).
inline auto equal_width_bin(float v, float lo, float hi, float inv_width,
                            int num_bins) -> int {
  if (std::isnan(v) || v < lo || v > hi)
    return -1;
  int b = (v == hi) ? num_bins - 1
                    : static_cast<int>((v - lo) * inv_width);
  return b >= num_bins ? num_bins - 1 : b;
}

/// @brief Bin index for explicit edges (length `num_bins + 1`, monotone).
/// Returns -1 for NaN or out-of-range values. `v == last_edge` lands in the
/// last bin (inclusive upper edge).
inline auto edges_bin(float v, tf::range<float *, tf::dynamic_size> edges,
                      int num_bins, float last_edge) -> int {
  if (std::isnan(v))
    return -1;
  auto it = std::upper_bound(edges.begin(), edges.end(), v);
  int b = static_cast<int>(it - edges.begin()) - 1;
  if (b < 0)
    return -1;
  if (b < num_bins)
    return b;
  return (v == last_edge) ? num_bins - 1 : -1;
}

// ============================================================================
// bincount — counts[v]++ for each v in x (optionally weighted).
// ============================================================================

/// @brief Counts occurrences of each non-negative integer.
///
/// Caller pre-sizes and zero-initializes `counts` (length >= max(x) + 1).
/// No bounds check; negatives or out-of-range values are UB.
inline auto bincount(tf::range<int *, tf::dynamic_size> x,
                     tf::range<int *, tf::dynamic_size> counts) -> void {
  for (auto v : x)
    ++counts[static_cast<std::size_t>(v)];
}

/// @brief Weighted bincount: `counts[x[i]] += w[i]`.
/// `x` and `w` must have the same length.
inline auto bincount(tf::range<int *, tf::dynamic_size> x,
                     tf::range<float *, tf::dynamic_size> w,
                     tf::range<float *, tf::dynamic_size> counts) -> void {
  auto xi = x.begin();
  auto wi = w.begin();
  auto xe = x.end();
  for (; xi != xe; ++xi, ++wi)
    counts[static_cast<std::size_t>(*xi)] += *wi;
}

// ============================================================================
// histogram_equal_width — bins over [lo, hi] inferred from counts.size().
// NaN and out-of-range values are dropped. Upper edge is inclusive.
// ============================================================================

/// @brief Integer counts.
inline auto histogram_equal_width(tf::range<float *, tf::dynamic_size> x,
                                  float lo, float hi,
                                  tf::range<int *, tf::dynamic_size> counts)
    -> void {
  int num_bins = static_cast<int>(counts.size());
  if (num_bins <= 0 || !(hi > lo))
    return;
  float inv_width = static_cast<float>(num_bins) / (hi - lo);
  for (auto v : x) {
    int b = equal_width_bin(v, lo, hi, inv_width, num_bins);
    if (b >= 0)
      ++counts[static_cast<std::size_t>(b)];
  }
}

/// @brief Float counts (used when the caller needs a float buffer, e.g.
/// pre-density). Equivalent to the integer variant semantically.
inline auto histogram_equal_width(tf::range<float *, tf::dynamic_size> x,
                                  float lo, float hi,
                                  tf::range<float *, tf::dynamic_size> counts)
    -> void {
  int num_bins = static_cast<int>(counts.size());
  if (num_bins <= 0 || !(hi > lo))
    return;
  float inv_width = static_cast<float>(num_bins) / (hi - lo);
  for (auto v : x) {
    int b = equal_width_bin(v, lo, hi, inv_width, num_bins);
    if (b >= 0)
      counts[static_cast<std::size_t>(b)] += 1.f;
  }
}

/// @brief Weighted float counts.
inline auto histogram_equal_width(tf::range<float *, tf::dynamic_size> x,
                                  tf::range<float *, tf::dynamic_size> w,
                                  float lo, float hi,
                                  tf::range<float *, tf::dynamic_size> counts)
    -> void {
  int num_bins = static_cast<int>(counts.size());
  if (num_bins <= 0 || !(hi > lo))
    return;
  float inv_width = static_cast<float>(num_bins) / (hi - lo);
  auto xi = x.begin();
  auto wi = w.begin();
  auto xe = x.end();
  for (; xi != xe; ++xi, ++wi) {
    int b = equal_width_bin(*xi, lo, hi, inv_width, num_bins);
    if (b >= 0)
      counts[static_cast<std::size_t>(b)] += *wi;
  }
}

// ============================================================================
// histogram_edges — explicit edges (length num_bins + 1, monotone).
// NaN and out-of-range values are dropped. Upper edge is inclusive.
// ============================================================================

/// @brief Integer counts.
inline auto histogram_edges(tf::range<float *, tf::dynamic_size> x,
                            tf::range<float *, tf::dynamic_size> edges,
                            tf::range<int *, tf::dynamic_size> counts) -> void {
  int num_bins = static_cast<int>(counts.size());
  if (num_bins <= 0)
    return;
  float last = edges[static_cast<std::size_t>(num_bins)];
  for (auto v : x) {
    int b = edges_bin(v, edges, num_bins, last);
    if (b >= 0)
      ++counts[static_cast<std::size_t>(b)];
  }
}

/// @brief Float counts.
inline auto histogram_edges(tf::range<float *, tf::dynamic_size> x,
                            tf::range<float *, tf::dynamic_size> edges,
                            tf::range<float *, tf::dynamic_size> counts)
    -> void {
  int num_bins = static_cast<int>(counts.size());
  if (num_bins <= 0)
    return;
  float last = edges[static_cast<std::size_t>(num_bins)];
  for (auto v : x) {
    int b = edges_bin(v, edges, num_bins, last);
    if (b >= 0)
      counts[static_cast<std::size_t>(b)] += 1.f;
  }
}

/// @brief Weighted float counts.
inline auto histogram_edges(tf::range<float *, tf::dynamic_size> x,
                            tf::range<float *, tf::dynamic_size> w,
                            tf::range<float *, tf::dynamic_size> edges,
                            tf::range<float *, tf::dynamic_size> counts)
    -> void {
  int num_bins = static_cast<int>(counts.size());
  if (num_bins <= 0)
    return;
  float last = edges[static_cast<std::size_t>(num_bins)];
  auto xi = x.begin();
  auto wi = w.begin();
  auto xe = x.end();
  for (; xi != xe; ++xi, ++wi) {
    int b = edges_bin(*xi, edges, num_bins, last);
    if (b >= 0)
      counts[static_cast<std::size_t>(b)] += *wi;
  }
}

// ============================================================================
// Support utilities.
// ============================================================================

/// @brief Build `num_bins + 1` uniformly-spaced edges. `edges[0] = lo`,
/// `edges[num_bins] = hi` exact (snapped to avoid FP drift).
inline auto make_linear_edges(float lo, float hi, int num_bins)
    -> tf::buffer<float> {
  tf::buffer<float> edges;
  edges.allocate(static_cast<std::size_t>(num_bins + 1));
  float width = (hi - lo) / static_cast<float>(num_bins);
  for (int i = 0; i <= num_bins; ++i)
    edges[static_cast<std::size_t>(i)] = lo + static_cast<float>(i) * width;
  edges[static_cast<std::size_t>(num_bins)] = hi;
  return edges;
}

/// @brief NaN-safe min/max. Returns `{+inf, -inf}` for an empty or
/// all-NaN range.
inline auto finite_min_max(tf::range<float *, tf::dynamic_size> x)
    -> std::pair<float, float> {
  float mn = std::numeric_limits<float>::infinity();
  float mx = -std::numeric_limits<float>::infinity();
  for (auto v : x) {
    if (std::isnan(v))
      continue;
    if (v < mn)
      mn = v;
    if (v > mx)
      mx = v;
  }
  return {mn, mx};
}

/// @brief Validate non-negativity and return `max(x)`. Returns -1 for empty
/// input. Throws `std::invalid_argument` on the first negative value.
inline auto max_nonneg_or_throw(tf::range<int *, tf::dynamic_size> x) -> int {
  int m = -1;
  for (auto v : x) {
    if (v < 0)
      throw std::invalid_argument(
          "bincount: input contains a negative value");
    if (v > m)
      m = v;
  }
  return m;
}

} // namespace ts
} // namespace tf

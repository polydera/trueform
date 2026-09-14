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

#include "../../int256.hpp"
#include "../../int32.hpp"
#include "../../int512.hpp"
#include "../../int64.hpp"
#include "../../meta.hpp"

namespace tf::exact::door::pool {

/// The two rungs an exact representative's arithmetic stands on.
///
/// A support normal is a cross of coordinate DIFFERENCES, so it occupies
/// twice a difference and not twice a coordinate — a rung above the one the
/// door's quantized names live on. The coefficient rung `R` holds it and the
/// offset it carries; the product rung `I` holds `R` against `R`, which is
/// where `N . N`, a squared residual and every frame product land.
///
/// `bound_bits` and `product_bits` are the ONE authority for both widths:
/// what a workspace component is narrowed inside, and the magnitude the
/// product rung carries. Every consumer that has to know a width reads it
/// here instead of restating a constant true at one lattice only.
template <typename Int> struct exact_lane;

template <> struct exact_lane<tf::exact::int32> {
  using coefficient_type = typename tf::exact::meta<tf::exact::int32>::T2;
  using product_type = tf::exact::int256;
  /// A workspace component is narrowed to `R` only inside `2^bound_bits`,
  /// so three of their products still stand inside `I`.
  static constexpr int bound_bits = 126;
  static constexpr int product_bits = 255;
};

template <> struct exact_lane<tf::exact::int64> {
  using coefficient_type = typename tf::exact::meta<tf::exact::int64>::T2;
  using product_type = tf::exact::int512;
  static constexpr int bound_bits = 254;
  static constexpr int product_bits = 511;
};

/// Whether the partition's arithmetic stands at this lattice. A lattice
/// with no lane has no certificate, so its door is the cascade alone —
/// and this is the one place that answers it, for the naming pass and the
/// placement alike.
template <typename Int> inline constexpr bool has_exact_lane = false;
template <> inline constexpr bool has_exact_lane<tf::exact::int32> = true;
template <> inline constexpr bool has_exact_lane<tf::exact::int64> = true;

} // namespace tf::exact::door::pool

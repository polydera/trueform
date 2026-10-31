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
#include "../../core/none.hpp"
#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace tf::geometry {

/// What a geodesic front cost: every point it took at the distance the point
/// still held, summed over the fronts that took them, and the most any one
/// front took.
struct geodesic_front_stats {
  /// Points taken at the distance they held, over every front.
  std::size_t pops = 0;
  /// The most points one front took.
  std::size_t largest = 0;
  /// The fronts that ran.
  std::size_t fronts = 0;
};

/// Count one front's `pops` into `stats`; nothing where `stats` is
/// `tf::none_t`, which is what a caller that did not ask passes.
template <typename Stats>
auto record_geodesic_front(Stats &stats, std::size_t pops) -> void {
  if constexpr (!std::is_same_v<std::decay_t<Stats>, tf::none_t>) {
    stats.pops += pops;
    stats.largest = std::max(stats.largest, pops);
    ++stats.fronts;
  }
}

} // namespace tf::geometry

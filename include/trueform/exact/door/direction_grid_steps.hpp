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

#include "../meta.hpp"

#include <limits>

namespace tf::exact::door {

/// THE GRAIN OF A DIRECTION CLASS, and the one producer of it. Its readers
/// are the pool tier's two questions about which way a face points: the
/// chart a name is pooled in (@ref
/// tf::exact::door::pool::pool_discovery_resolution) and the flatness a
/// vertex is read by (@ref tf::exact::door::pool::find_scene_features).
///
/// A CLASS IS DIMENSIONLESS, so the pitch its grid is stepped by must not
/// follow the lattice. The band does: it is a lattice distance, so the same
/// physical tolerance is a handful of units on a 32-bit lattice and billions
/// on a 64-bit one. Stepping a class by the raw band would make the grid
/// finer as the lattice widens, until no two neighbouring faces round onto
/// one direction and every vertex reads as a feature.
///
/// The band is stated in the canonical 32-bit grain instead: a `w`-bit
/// lattice shifts it down by `w - 32`, the identity at 32. One step is the
/// coarsest grid the door already runs at a small band, so the floor keeps
/// the existing regime.
///
/// A class is a grouping and never an identity: what a vertex lands on is
/// an exact plane, so the pitch decides only which names may pool.
template <typename Int>
auto direction_grid_steps(typename tf::exact::meta<Int>::T1 band) ->
    typename tf::exact::meta<Int>::T1 {
  using T1 = typename tf::exact::meta<Int>::T1;
  const T1 steps = band >> unsigned(std::numeric_limits<Int>::digits + 1 - 32);
  return steps > T1(0) ? steps : T1(1);
}

} // namespace tf::exact::door

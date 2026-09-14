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
/// THE DOOR'S OWN NAME IS QUANTIZED AT THE RAW BAND and not here (@ref
/// tf::exact::door::quantize_face_plane). That grid is POSITIONAL — the
/// finer it is the closer the named plane stands to the face, and its
/// offset is a lattice distance on the same grid — so it keeps the band
/// whole, exactly as an offset step does (@ref
/// tf::exact::door::plane_step). The two pitches coincide at int32, where
/// the shift is zero, and stand `2^32` apart at int64; nothing requires
/// them to agree, because a name is an identity and a class is a grouping.
template <typename Int>
auto direction_grid_steps(typename tf::exact::meta<Int>::T1 band) ->
    typename tf::exact::meta<Int>::T1 {
  using T1 = typename tf::exact::meta<Int>::T1;
  const T1 steps = band >> unsigned(std::numeric_limits<Int>::digits + 1 - 32);
  return steps > T1(0) ? steps : T1(1);
}

} // namespace tf::exact::door

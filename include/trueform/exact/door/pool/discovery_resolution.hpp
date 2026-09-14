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

#include "../direction_grid_steps.hpp"

#include "../../meta.hpp"

namespace tf::exact::door::pool {

/// THE DISCOVERY RESOLUTION: the pitch of the projective chart the
/// partition groups directions by.
///
/// It is the SAME PITCH the tier's flatness test reads, from the same
/// producer — @ref tf::exact::door::direction_grid_steps — so a vertex the
/// scene calls flat and the cell its faces fall in cannot disagree about
/// which two faces point the same way. It is NOT the door's plane-naming
/// pitch, which steps a direction by the raw band and stands `2^32` finer at
/// int64; a name is an identity and a cell is a grouping. Four is the floor,
/// below which a cell spans too much of the sphere to mean anything.
///
/// A caller-defined `TF_POOL_DISCOVERY_K` replaces it with a
/// scene-independent chart.
template <typename Int>
auto pool_discovery_resolution(typename tf::exact::meta<Int>::T1 band) -> int {
#ifdef TF_POOL_DISCOVERY_K
  (void)band;
  return TF_POOL_DISCOVERY_K;
#else
  // the band is capped a quarter below the lattice's own range, so the
  // grain-normalized step is an int at either width
  const int steps = int(tf::exact::door::direction_grid_steps<Int>(band));
  return steps > 4 ? steps : 4;
#endif
}

} // namespace tf::exact::door::pool

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

#include "../../meta.hpp"
#include "../round_to_wide.hpp"

namespace tf::exact::door::pool {

/// The bound a narrow-lane workspace component is admitted inside.
///
/// It is a QUARTER of @ref tf::exact::door::wide_placement_bound, which is
/// where the door's own steps refuse. A Lagrange step and a Gram step each
/// STATE a candidate before they know it is shorter, and such a candidate
/// stands at most two and a half times the components it was formed from, so
/// only a component with that headroom cannot reach the guarded steps' own
/// refusal.
///
/// WHAT THE TWO LANES SHARE IS ADMISSIBILITY, NOT THE POINT. Each reduces
/// its frame on its own rung — the narrow one screens a double-rounded
/// quotient against a wide dot, the wide one reduces on the product rung —
/// so a direction both lanes could state may come back as different lattice
/// points of the same plane. Every answer of either lane lies ON that plane
/// and inside the band, which is the whole of what the certificate asks; the
/// lane moves the price and which admissible point is returned.
template <typename Int>
auto narrow_lane_bound() -> typename tf::exact::meta<Int>::T1 {
  return tf::exact::door::wide_placement_bound<Int>() >> 2;
}

} // namespace tf::exact::door::pool

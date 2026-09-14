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

#include "./exact_lane.hpp"

namespace tf::exact::door::pool {

/// The bound every exact workspace component is held inside before it is
/// stated on the coefficient rung. It is not
/// @ref tf::exact::door::wide_placement_bound, which bounds the door's
/// quantized lane a rung below; a coefficient outside this one has no
/// frame and its vertex has no certificate.
template <typename Int>
auto exact_placement_bound() -> typename exact_lane<Int>::coefficient_type {
  using coefficient_type = typename exact_lane<Int>::coefficient_type;
  return coefficient_type(1) << unsigned(exact_lane<Int>::bound_bits);
}

} // namespace tf::exact::door::pool

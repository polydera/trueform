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

#include "./diametral_admits.hpp"
#include "./meta.hpp"

namespace tf::exact {

/// Bits of uniform scale above the `Int` lattice the dot still admits.
template <typename Int> constexpr auto diametral_scale_bits() -> int {
  int scale = 0;
  while (diametral_admits<Int>(meta<Int>::coordinate_bits + scale + 1))
    ++scale;
  return scale;
}

} // namespace tf::exact

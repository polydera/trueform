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

#include "./diametral_bits.hpp"
#include "./meta.hpp"

namespace tf::exact {

/// Whether the differences hold on `meta<Int>::T1` and the dot on
/// `meta<Int>::T2`.
template <typename Int>
constexpr auto diametral_admits(int lattice_bits) -> bool {
  return lattice_bits + 1 <= meta<Int>::t1_bits &&
         diametral_bits(lattice_bits) <= meta<Int>::t2_bits;
}

static_assert(diametral_admits<int32>(meta<int32>::coordinate_bits),
              "the int32 lattice must fit its own diametral dot");
static_assert(diametral_admits<int64>(meta<int64>::coordinate_bits),
              "the int64 lattice must fit its own diametral dot");

} // namespace tf::exact

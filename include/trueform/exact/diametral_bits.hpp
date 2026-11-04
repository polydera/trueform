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

namespace tf::exact {

/// Magnitude bits of `(a - s) . (b - s)` over a lattice of `lattice_bits`
/// value bits. A difference spans one bit more than a coordinate, and the
/// dot is three of their products: below `3 * 2^(2(b + 1))`, every partial
/// sum below `2^(2(b + 1) + 1)`.
constexpr auto diametral_bits(int lattice_bits) -> int {
  return 2 * (lattice_bits + 1) + 2;
}

} // namespace tf::exact

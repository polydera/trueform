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

#include "./diametral_value.hpp"
#include "./vertex.hpp"

namespace tf::exact {

/// Exact sign of @ref tf::exact::diametral_value. Returns -1, 0, or +1.
/// Unfiltered: the exact path is one double-width dot.
template <typename Int, typename Coord>
auto diametral_sign(const pt3<Coord> &a, const pt3<Coord> &b,
                    const pt3<Coord> &s) -> int {
  const auto value = diametral_value<Int>(a, b, s);
  return (value > 0) ? 1 : (value < 0) ? -1 : 0;
}

} // namespace tf::exact

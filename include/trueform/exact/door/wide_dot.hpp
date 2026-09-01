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

namespace tf::exact::door {

/// The dot product of two lattice triples, on the rung above them.
template <typename Int, typename A, typename B>
auto wide_dot(const A &a, const B &b) -> typename tf::exact::meta<Int>::T2 {
  using T2 = typename tf::exact::meta<Int>::T2;
  return T2(a[0]) * T2(b[0]) + T2(a[1]) * T2(b[1]) + T2(a[2]) * T2(b[2]);
}

} // namespace tf::exact::door

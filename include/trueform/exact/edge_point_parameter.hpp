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

#include "./edge_parameter.hpp"
#include "./meta.hpp"
#include "./vertex.hpp"

namespace tf::exact {

/// The exact parameter of a point `q` that lies on the line of
/// (p0, p1): num = <q - p0, d>, den = <d, d> with d = p1 - p0. Caller
/// states the incidence — a record that names `q` as a vertex on that
/// edge is that statement — and the fraction is then exactly q's
/// position, since q = p0 + (num/den) d follows from collinearity.
///
/// Degree 2: coordinate differences are T1, their products and the
/// three-term sum are T2 with room to spare (68 value bits for an
/// int32 lattice, 132 for int64). `den > 0` for any p0 != p1.
template <typename Int>
auto make_edge_point_parameter(const pt3<Int> &p0, const pt3<Int> &p1,
                               const pt3<Int> &q) -> edge_parameter<Int> {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  const T1 dx = T1(p1[0]) - p0[0], dy = T1(p1[1]) - p0[1],
           dz = T1(p1[2]) - p0[2];
  const T1 wx = T1(q[0]) - p0[0], wy = T1(q[1]) - p0[1], wz = T1(q[2]) - p0[2];
  const T2 num = T2(wx) * dx + T2(wy) * dy + T2(wz) * dz;
  const T2 den = T2(dx) * dx + T2(dy) * dy + T2(dz) * dz;
  return {num, den};
}

} // namespace tf::exact

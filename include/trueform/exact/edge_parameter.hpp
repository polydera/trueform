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

#include "./det2_sign.hpp"
#include "./meta.hpp"

namespace tf::exact {

/// A point on a segment carrier as an exact fraction of the carrier's
/// span: position = num / den along (p0 -> p1), den > 0 by
/// construction. Nothing is divided and nothing is rounded — the pair
/// is the decision currency; a lattice position is materialized once,
/// at the canonical point, via @ref tf::exact::dyadic_ratio.
///
/// A parameter outside [0, den] is a valid line position: carriers are
/// lines, an edge is an interval on one.
template <typename Int> struct edge_parameter {
  typename meta<Int>::T2 num;
  typename meta<Int>::T2 den;
};

/// Order of two parameters on one carrier: sign(x - y). Over a shared
/// span the difference is `(x.num - y.num) / den`, so the sign is the
/// numerators' — the determinant is only consulted across spans.
template <typename Int>
auto compare_parameter(const edge_parameter<Int> &x,
                       const edge_parameter<Int> &y) -> int {
  if (x.den == y.den)
    return x.num == y.num ? 0 : (x.num < y.num ? -1 : 1);
  return det2_sign<Int>(x.num, y.den, y.num, x.den);
}

/// The same position, measured from the carrier's other endpoint. Every
/// parameter construction — plane crossing, edge crossing, point on a
/// line — reverses to exactly this pair, so a carrier's direction is a
/// question of naming, never of recomputation.
template <typename Int>
auto reversed_parameter(const edge_parameter<Int> &t) -> edge_parameter<Int> {
  return {t.den - t.num, t.den};
}

} // namespace tf::exact

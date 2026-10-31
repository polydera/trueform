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

#include "./plane_edge_radial_authority.hpp"
#include "./plane_pair_carrier.hpp"

namespace tf::intersect::graph {

/// The sign of a valid authority's line against `normal`, a vector of degree
/// two or less: zero exactly when a plane of that normal runs along the line.
template <typename Int, typename Vector>
auto plane_edge_radial_line_dot_sign(
    const plane_edge_radial_authority<Int> &authority, const Vector &normal)
    -> int {
  if (authority.carrier.valid)
    return plane_carrier_dot_sign<Int>(authority.carrier, normal);
  const auto dot = plane_carrier_dot<Int>(normal, authority.edge);
  return (dot > 0) - (dot < 0);
}

} // namespace tf::intersect::graph

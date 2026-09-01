/*
 * Copyright (c) 2025 XLAB
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
#include <array>
#include <cstddef>

namespace tf::topology::cdt {

/// The input constraint each edge of face `triangle` belongs to and the span
/// it covers, oriented from corner k to corner k + 1. A refinement sub-edge
/// carries its origin through every split, so a split piece states its
/// original input constraint on that constraint's whole scale.
template <typename Owner>
auto constrained_delaunay_refinement_face_constraint_owners(
    const Owner &owner, typename Owner::index_type triangle)
    -> std::array<typename Owner::constraint_provenance, 3> {
  using Param = typename Owner::param_type;
  using Provenance = typename Owner::constraint_provenance;

  const auto &face = owner._t[std::size_t(triangle)];
  const auto provenance = [&](int edge) -> Provenance {
    const auto segment = face.seg[edge];
    if (segment == Owner::none)
      return {Owner::none, Param(0), Param(0)};
    const auto &sub = owner._segments[std::size_t(segment)];
    const int shift = Owner::crossing_param_bits - int(sub.d);
    const Param low = Param(sub.m) << shift;
    const Param high = (Param(sub.m) + Param(1)) << shift;
    return face.v[edge] == sub.lo_vertex ? Provenance{sub.origin, low, high}
                                         : Provenance{sub.origin, high, low};
  };
  return {provenance(0), provenance(1), provenance(2)};
}

} // namespace tf::topology::cdt

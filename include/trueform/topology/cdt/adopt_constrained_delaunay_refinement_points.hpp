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
#include "../../core/point.hpp"
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto adopt_constrained_delaunay_refinement_points(Owner &owner) -> void {
  using Int = typename Owner::int_type;

  auto out = owner._cdt.converted_points();
  auto ints = owner._cdt.points();
  owner._ip.reserve(ints.size());
  owner._dp.reserve(ints.size());
  owner._con_nbr.reserve(ints.size());
  for (std::size_t i = 0; i < std::size_t(ints.size()); ++i) {
    owner._ip.push_back(tf::point<Int, 2>{ints[i][0], ints[i][1]});
    owner._dp.push_back({double(out[i][0]), double(out[i][1])});
    owner._con_nbr.push_back({Owner::none, Owner::none});
  }

  // Affine double->int from the x-extreme pair (two arbitrary points risk
  // catastrophic cancellation when nearly x-coincident).
  std::size_t lo = 0, hi = 0;
  for (std::size_t i = 1; i < owner._dp.size(); ++i) {
    if (owner._dp[i][0] < owner._dp[lo][0])
      lo = i;
    if (owner._dp[i][0] > owner._dp[hi][0])
      hi = i;
  }
  double dx = owner._dp[hi][0] - owner._dp[lo][0];
  owner._scale =
      dx != 0 ? double(owner._ip[hi][0] - owner._ip[lo][0]) / dx : 1.0;
  owner._offx = double(owner._ip[lo][0]) - owner._dp[lo][0] * owner._scale;
  owner._offy = double(owner._ip[lo][1]) - owner._dp[lo][1] * owner._scale;
}

} // namespace tf::topology::cdt

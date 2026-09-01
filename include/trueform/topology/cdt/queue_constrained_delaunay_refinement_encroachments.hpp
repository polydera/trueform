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
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto queue_constrained_delaunay_refinement_encroachments(
    Owner &owner, typename Owner::index_type face) -> bool {
  using Index = typename Owner::index_type;

  bool any = false;
  for (int edge = 0; edge < 3; ++edge) {
    if (!owner.constrained(face, edge) || !owner.splittable(face, edge))
      continue;
    Index u = owner._t[face].v[edge];
    Index v = owner._t[face].v[(edge + 1) % 3];
    Index w = owner._t[face].v[(edge + 2) % 3];
    if (owner.constraint_connected(w, u) || owner.constraint_connected(w, v))
      continue;
    if (owner.encroaches(owner._dp[std::size_t(u)],
                         owner._dp[std::size_t(v)],
                         owner._dp[std::size_t(w)])) {
      owner._pending.push_back({face, Index(edge), owner._t[face].stamp});
      any = true;
    }
  }
  return any;
}

} // namespace tf::topology::cdt

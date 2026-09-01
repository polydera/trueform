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
#include <utility>

namespace tf::topology::cdt {

template <typename Owner>
auto locate_constrained_delaunay_refinement_point(
    const Owner &owner, typename Owner::index_type from,
    typename Owner::index_type point)
    -> std::pair<typename Owner::index_type, int> {
  using Index = typename Owner::index_type;

  Index face = from;
  for (int guard = 0; guard < 1024; ++guard) {
    int exit_edge = -1;
    for (int edge = 0; edge < 3; ++edge)
      if (owner.orient(owner._t[face].v[edge],
                       owner._t[face].v[(edge + 1) % 3], point) < 0) {
        exit_edge = edge;
        break;
      }
    if (exit_edge < 0)
      return {face, -1};
    if (owner.constrained(face, exit_edge))
      return {face, exit_edge};
    Index next = owner._t[face].n[exit_edge];
    if (next == Owner::none)
      return {Owner::none, -1};
    face = next;
  }
  return {Owner::none, -1};
}

} // namespace tf::topology::cdt

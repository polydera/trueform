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
#include "./for_each_constrained_delaunay_outgoing.hpp"

namespace tf::topology::cdt {

template <typename Owner>
auto find_constrained_delaunay_edge(const Owner &owner,
                                    typename Owner::index_type first_vertex,
                                    typename Owner::index_type second_vertex) ->
    typename Owner::index_type {
  using Index = typename Owner::index_type;
  Index found = Owner::none;
  for_each_constrained_delaunay_outgoing(owner, first_vertex, [&](Index edge) {
    if (owner.target(edge) != second_vertex)
      return true;
    found = edge;
    return false;
  });
  return found;
}

} // namespace tf::topology::cdt

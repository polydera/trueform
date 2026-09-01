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

template <typename Owner, typename F>
auto for_each_constrained_delaunay_outgoing(const Owner &owner,
                                            typename Owner::index_type vertex,
                                            F &&function) -> void {
  using Index = typename Owner::index_type;
  const Index first = owner._v_first_edge[std::size_t(vertex)];
  if (first == Owner::none)
    return;

  Index edge = first;
  do {
    if (!function(edge))
      return;
    edge = owner.next_edge(edge);
  } while (edge != first);
}

} // namespace tf::topology::cdt

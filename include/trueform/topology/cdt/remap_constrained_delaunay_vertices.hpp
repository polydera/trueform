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
auto remap_constrained_delaunay_vertices(Owner &owner) -> void {
  if (owner._sites.size() == 0)
    return;
  for (std::size_t i = 0; i < owner._edges.size(); ++i) {
    auto edge = owner._edges[i];
    if (edge.vertex != Owner::none)
      edge.vertex = owner._sites[std::size_t(edge.vertex)].output;
  }
  owner._sites.clear();
  owner._keys.clear();
}

} // namespace tf::topology::cdt

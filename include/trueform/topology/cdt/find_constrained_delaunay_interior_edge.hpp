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
auto find_constrained_delaunay_interior_edge(const Owner &owner) ->
    typename Owner::index_type {
  using Index = typename Owner::index_type;
  for (std::size_t edge = 0; edge < owner._edges.size(); ++edge)
    if (!owner._edges[edge].boundary)
      return static_cast<Index>(edge);
  return Owner::none;
}

} // namespace tf::topology::cdt

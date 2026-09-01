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
#include "./find_constrained_delaunay_interior_edge.hpp"
#include "./size_adaptive.hpp"
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto rebuild_constrained_delaunay_vertex_edges(Owner &owner,
                                               std::size_t n_vertices) -> void {
  using Index = typename Owner::index_type;
  owner._v_first_edge.allocate(n_vertices);
  topology::cdt::fill_auto(owner._v_first_edge, Owner::none);
  for (std::size_t edge = 0; edge < owner._edges.size(); ++edge) {
    const Index vertex = owner._edges[edge].vertex;
    if (vertex != Owner::none &&
        owner._v_first_edge[std::size_t(vertex)] == Owner::none)
      owner._v_first_edge[std::size_t(vertex)] = static_cast<Index>(edge);
  }
  owner._locate_hint = find_constrained_delaunay_interior_edge(owner);
}

} // namespace tf::topology::cdt

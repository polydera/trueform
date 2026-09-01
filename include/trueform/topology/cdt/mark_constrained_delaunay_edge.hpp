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
auto mark_constrained_delaunay_edge(Owner &owner,
                                    typename Owner::index_type edge,
                                    bool is_boundary) -> void {
  const auto apply = [&](typename Owner::index_type half_edge) {
    auto &state = owner._edges[std::size_t(half_edge)].constrained;
    if (is_boundary)
      state = state == Owner::boundary_constrained
                  ? Owner::constrained
                  : Owner::boundary_constrained;
    else if (state == Owner::unconstrained)
      state = Owner::constrained;
  };
  apply(edge);
  apply(Owner::opposite(edge));
  owner._edges[std::size_t(edge)].delaunay = true;
  owner._edges[std::size_t(Owner::opposite(edge))].delaunay = true;
}

} // namespace tf::topology::cdt

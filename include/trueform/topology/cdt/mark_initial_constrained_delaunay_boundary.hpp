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
#include <initializer_list>

namespace tf::topology::cdt {

template <typename Owner>
auto mark_initial_constrained_delaunay_boundary(Owner &owner,
                                                typename Owner::index_type edge)
    -> void {
  for (const auto half_edge : {edge, Owner::opposite(edge)}) {
    owner._edges[std::size_t(half_edge)].boundary = true;
    owner._edges[std::size_t(half_edge)].delaunay = true;
  }
}

} // namespace tf::topology::cdt

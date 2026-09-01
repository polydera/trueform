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
auto queue_constrained_delaunay_flip(Owner &owner,
                                     typename Owner::index_type edge,
                                     typename Owner::index_type opposite_vertex)
    -> void {
  owner._edges[std::size_t(edge)].delaunay = false;
  owner._edges[std::size_t(Owner::opposite(edge))].delaunay = false;
  owner._flip_stack.push_back(typename Owner::flip_check{
      edge, owner.origin(edge), owner.target(edge), opposite_vertex});
}

} // namespace tf::topology::cdt

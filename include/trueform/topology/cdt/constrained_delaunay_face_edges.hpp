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
#include <array>
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto constrained_delaunay_face_edges(const Owner &owner,
                                     typename Owner::index_type triangle)
    -> std::array<typename Owner::index_type, 3> {
  using Index = typename Owner::index_type;
  const Index first = owner._first_edge_of_triangle[std::size_t(triangle)];
  const Index second = owner.previous_edge(Owner::opposite(first));
  const Index third = owner.previous_edge(Owner::opposite(second));
  return {first, second, third};
}

} // namespace tf::topology::cdt

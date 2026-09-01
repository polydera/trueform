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
#include "./constrained_delaunay_face_edges.hpp"
#include "./size_adaptive.hpp"
#include <cstddef>

namespace tf::topology::cdt {

/// The canonical dart of each face is the face materializer's product; this
/// phase adds only the inverse, the face each dart bounds.
template <typename Owner>
auto materialize_constrained_delaunay_face_adjacency(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  auto &triangle_of_edge = owner._triangle_of_edge;
  triangle_of_edge.allocate(owner._edges.size());
  topology::cdt::fill_auto(triangle_of_edge, Owner::none);

  const auto &first_edges = owner._first_edge_of_triangle;
  for (Index triangle = 0; triangle < static_cast<Index>(first_edges.size());
       ++triangle) {
    const auto edges = constrained_delaunay_face_edges(owner, triangle);
    triangle_of_edge[std::size_t(edges[0])] = triangle;
    triangle_of_edge[std::size_t(edges[1])] = triangle;
    triangle_of_edge[std::size_t(edges[2])] = triangle;
  }
}

} // namespace tf::topology::cdt

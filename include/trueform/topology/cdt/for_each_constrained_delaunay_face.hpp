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
#include <array>
#include <cassert>
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner, typename F>
auto for_each_constrained_delaunay_face(const Owner &owner, F &&function)
    -> void {
  using Index = typename Owner::index_type;
  const auto &labels = owner._region_labels;
  assert(labels.size() == std::size_t(owner._faces.size()));
  assert(owner._first_edge_of_triangle.size() ==
         std::size_t(owner._faces.size()));
  const auto neighbor = [&](Index edge) {
    return owner._triangle_of_edge[std::size_t(Owner::opposite(edge))];
  };
  const auto constrained = [&](Index edge) {
    return owner._edges[std::size_t(edge)].constrained != Owner::unconstrained;
  };
  for (Index triangle = 0;
       triangle < static_cast<Index>(owner._first_edge_of_triangle.size());
       ++triangle) {
    const auto edges = constrained_delaunay_face_edges(owner, triangle);
    const auto vertices = owner._faces[std::size_t(triangle)];
    function(triangle, vertices[0], vertices[1], vertices[2],
             labels[std::size_t(triangle)],
             std::array<Index, 3>{neighbor(edges[0]), neighbor(edges[1]),
                                  neighbor(edges[2])},
             std::array<bool, 3>{constrained(edges[0]), constrained(edges[1]),
                                 constrained(edges[2])});
  }
}

} // namespace tf::topology::cdt

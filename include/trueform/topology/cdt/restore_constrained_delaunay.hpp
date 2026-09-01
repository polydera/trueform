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
#include "./flip_constrained_delaunay_edge.hpp"
#include "./queue_constrained_delaunay_flip.hpp"
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto restore_constrained_delaunay(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  while (owner._flip_stack.size() != 0) {
    const auto check = owner._flip_stack.back();
    owner._flip_stack.pop_back();
    const Index edge = check.e;
    const Index opposite = Owner::opposite(edge);
    if (owner.origin(edge) != check.v0 || owner.target(edge) != check.v1)
      continue;
    if (owner._edges[std::size_t(edge)].delaunay ||
        owner._edges[std::size_t(opposite)].delaunay ||
        owner._edges[std::size_t(edge)].constrained ||
        owner._edges[std::size_t(opposite)].constrained)
      continue;
    if (owner._edges[std::size_t(edge)].boundary ||
        owner._edges[std::size_t(opposite)].boundary) {
      owner._edges[std::size_t(edge)].delaunay = true;
      owner._edges[std::size_t(opposite)].delaunay = true;
      continue;
    }

    const Index vertex0 = owner.origin(edge);
    const Index vertex1 = owner.origin(opposite);
    const Index other0 = owner.origin(Owner::opposite(owner.next_edge(edge)));
    const Index other1 =
        owner.origin(Owner::opposite(owner.next_edge(opposite)));
    if (other0 == other1)
      continue;
    if (owner.incircle(vertex0, vertex1, other1, other0) > 0) {
      flip_constrained_delaunay_edge(owner, edge);
      if (check.opposite_vertex == other0) {
        queue_constrained_delaunay_flip(owner, owner.previous_edge(edge),
                                        check.opposite_vertex);
        queue_constrained_delaunay_flip(owner, owner.next_edge(edge),
                                        check.opposite_vertex);
      } else {
        queue_constrained_delaunay_flip(owner, owner.next_edge(opposite),
                                        check.opposite_vertex);
        queue_constrained_delaunay_flip(owner, owner.previous_edge(opposite),
                                        check.opposite_vertex);
      }
    }
    owner._edges[std::size_t(edge)].delaunay = true;
    owner._edges[std::size_t(opposite)].delaunay = true;
  }
}

} // namespace tf::topology::cdt

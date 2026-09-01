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
#include "./for_each_constrained_delaunay_outgoing.hpp"
#include "./is_between_constrained_delaunay_vertices.hpp"

namespace tf::topology::cdt {

/// Find the first triangle crossed by a constraint leaving `first_vertex`.
/// `blocking` names an existing vertex strictly on the constraint when that is
/// what prevents the walk from starting, preserving a resolvable obstruction.
template <typename Owner>
auto find_initial_constrained_delaunay_triangle(
    const Owner &owner, typename Owner::index_type first_vertex,
    typename Owner::index_type second_vertex,
    typename Owner::index_type &initial,
    typename Owner::index_type &clockwise_vertex,
    typename Owner::index_type &counterclockwise_vertex,
    typename Owner::index_type *blocking = nullptr) -> bool {
  using Index = typename Owner::index_type;
  bool found = false;
  bool valid = true;
  for_each_constrained_delaunay_outgoing(owner, first_vertex, [&](Index edge) {
    const Index previous_vertex =
        owner.origin(Owner::opposite(owner.next_edge(edge)));
    const Index next_vertex = owner.target(edge);
    const int next_orientation =
        owner.orient(first_vertex, second_vertex, next_vertex);
    const int previous_orientation =
        owner.orient(first_vertex, second_vertex, previous_vertex);

    if (previous_orientation == 0 || next_orientation == 0) {
      const Index collinear =
          previous_orientation == 0 ? previous_vertex : next_vertex;
      if (is_between_constrained_delaunay_vertices(owner, first_vertex,
                                                   second_vertex, collinear)) {
        valid = false;
        found = true;
        if (blocking != nullptr)
          *blocking = collinear;
        return false;
      }
      return true;
    }
    if (previous_orientation < 0 && next_orientation > 0) {
      initial = edge;
      clockwise_vertex = previous_vertex;
      counterclockwise_vertex = next_vertex;
      found = true;
      return false;
    }
    return true;
  });
  return found && valid;
}

} // namespace tf::topology::cdt

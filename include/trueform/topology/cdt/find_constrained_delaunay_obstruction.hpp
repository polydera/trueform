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
#include "./find_initial_constrained_delaunay_triangle.hpp"
#include <cstddef>

namespace tf::topology::cdt {

/// Report the first constrained edge or existing vertex met while walking from
/// `first_vertex` toward `second_vertex`, without mutating the triangulation.
template <typename Owner>
auto find_constrained_delaunay_obstruction(
    const Owner &owner, typename Owner::index_type first_vertex,
    typename Owner::index_type second_vertex) -> typename Owner::obstruction {
  using Index = typename Owner::index_type;
  using ObstructionKind = typename Owner::obstruction_kind;
  Index initial = Owner::none;
  Index clockwise_vertex = Owner::none;
  Index counterclockwise_vertex = Owner::none;
  Index blocking = Owner::none;
  if (!find_initial_constrained_delaunay_triangle(
          owner, first_vertex, second_vertex, initial, clockwise_vertex,
          counterclockwise_vertex, &blocking)) {
    if (blocking != Owner::none)
      return {ObstructionKind::vertex_on, Owner::none, blocking};
    return {};
  }

  Index edge = owner.previous_edge(Owner::opposite(initial));
  // A corrupt fan must terminate as a refusal rather than spin indefinitely.
  for (std::size_t guard = 0; guard < owner._edges.size() + 2; ++guard) {
    if (owner._edges[std::size_t(edge)].constrained != Owner::unconstrained) {
      const Index first = owner.origin(edge);
      const Index second = owner.target(edge);
      if (owner.orient(first_vertex, second_vertex, first) == 0 &&
          owner.orient(first_vertex, second_vertex, second) == 0)
        return {ObstructionKind::collinear, edge, Owner::none};
      return {ObstructionKind::crossing, edge, Owner::none};
    }
    const Index third_edge = Owner::opposite(owner.previous_edge(edge));
    const Index third_vertex = owner.origin(third_edge);
    if (third_vertex == second_vertex)
      return {};
    const int orientation =
        owner.orient(first_vertex, second_vertex, third_vertex);
    if (orientation == 0)
      return {ObstructionKind::vertex_on, Owner::none, third_vertex};
    edge = orientation < 0 ? Owner::opposite(third_edge)
                           : owner.previous_edge(third_edge);
  }
  return {};
}

} // namespace tf::topology::cdt

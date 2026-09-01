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
#include "./unlink_constrained_delaunay_edge.hpp"
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto remove_constrained_delaunay_cavity(
    Owner &owner, typename Owner::index_type first_vertex,
    typename Owner::index_type second_vertex,
    typename Owner::index_type clockwise_vertex,
    typename Owner::index_type counterclockwise_vertex,
    typename Owner::index_type initial,
    typename Owner::index_type *passed_through = nullptr) -> bool {
  using Index = typename Owner::index_type;
  owner._vertices_cw.push_back(first_vertex);
  owner._vertices_cw.push_back(clockwise_vertex);
  owner._vertices_ccw.push_back(first_vertex);
  owner._vertices_ccw.push_back(counterclockwise_vertex);

  Index across = owner.previous_edge(Owner::opposite(initial));
  bool reached_second = false;
  // Constraint recovery mutates the fan while walking it. Bound the walk so a
  // corrupt fan refuses cleanly and leaves retry policy to the recovery phase.
  for (std::size_t guard = 0;
       guard < owner._edges.size() + 2 && !reached_second; ++guard) {
    if (owner._edges[std::size_t(across)].constrained)
      return false;

    const Index third_edge = Owner::opposite(owner.previous_edge(across));
    const Index third_vertex = owner.origin(third_edge);
    Index next_across = Owner::none;
    const bool reached = third_vertex == second_vertex;
    if (!reached) {
      const int orientation =
          owner.orient(first_vertex, second_vertex, third_vertex);
      if (orientation == 0) {
        // PASSAGE IS INCIDENCE. An exact zero says this vertex lies ON the
        // constraint's interior, so the corridor has no interior here and
        // no cavity can be carved. The constraint is not unrecoverable, it
        // is UNSPLIT — it must own this vertex, and the caller states that
        // split so the next build carves two well-formed cavities. The
        // radius is the ROUNDING UNIT, not a tolerance: this is exact
        // integer orientation and it fires at tolerance zero.
        if (passed_through != nullptr)
          *passed_through = third_vertex;
        return false;
      }
      if (orientation < 0) {
        clockwise_vertex = third_vertex;
        owner._vertices_cw.push_back(third_vertex);
        next_across = Owner::opposite(third_edge);
      } else {
        counterclockwise_vertex = third_vertex;
        owner._vertices_ccw.push_back(third_vertex);
        next_across = owner.previous_edge(third_edge);
      }
    }

    unlink_constrained_delaunay_edge(owner, across);
    owner._deleted_edges.push_back(across);
    if (reached) {
      reached_second = true;
      break;
    }
    across = next_across;
  }
  if (!reached_second)
    return false;

  owner._vertices_cw.push_back(second_vertex);
  owner._vertices_ccw.push_back(second_vertex);
  return owner._vertices_cw.size() >= 3 && owner._vertices_ccw.size() >= 3;
}

} // namespace tf::topology::cdt

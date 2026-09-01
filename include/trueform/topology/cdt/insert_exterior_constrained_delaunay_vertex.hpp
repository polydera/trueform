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
#include "./append_constrained_delaunay_edge.hpp"
#include "./mark_initial_constrained_delaunay_boundary.hpp"
#include "./queue_constrained_delaunay_flip.hpp"
#include "./restore_constrained_delaunay.hpp"
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto insert_exterior_constrained_delaunay_vertex(
    Owner &owner, typename Owner::index_type vertex) -> void {
  using Index = typename Owner::index_type;
  const auto is_visible = [&](Index edge) {
    return owner.orient(owner.origin(edge), owner.target(edge), vertex) > 0;
  };
  const auto next_boundary = [&](Index edge) {
    return owner.next_edge(Owner::opposite(edge));
  };
  const auto previous_boundary = [&](Index edge) {
    return Owner::opposite(owner.previous_edge(edge));
  };

  Index last_visible_forward = owner._last_edge;
  Index last_visible_backward = previous_boundary(owner._last_edge);
  Index previous_visible_backward = owner._last_edge;
  while (is_visible(last_visible_forward))
    last_visible_forward = next_boundary(last_visible_forward);
  while (is_visible(last_visible_backward)) {
    previous_visible_backward = last_visible_backward;
    last_visible_backward = previous_boundary(last_visible_backward);
  }

  if (last_visible_forward == previous_visible_backward) {
    // A collinear prefix has no face, so extend only its hull chain.
    const Index previous_vertex = vertex - Index(1);
    owner._last_edge = append_constrained_delaunay_edge(
        owner, vertex, previous_vertex, Owner::none, owner._last_edge, true);
    mark_initial_constrained_delaunay_boundary(owner, owner._last_edge);
    return;
  }

  Index current = previous_visible_backward;
  Index last_added = Owner::none;
  while (current != last_visible_forward) {
    const Index next = next_boundary(current);
    const Index first_added =
        last_added != Owner::none
            ? last_added
            : append_constrained_delaunay_edge(
                  owner, vertex, owner.origin(current), Owner::none,
                  owner.previous_edge(current), false);
    const Index second_added = append_constrained_delaunay_edge(
        owner, vertex, owner.target(current), owner.previous_edge(first_added),
        Owner::opposite(current), false);
    owner._edges[std::size_t(Owner::opposite(current))].boundary = false;
    if (last_added == Owner::none)
      owner._edges[std::size_t(first_added)].boundary = true;
    queue_constrained_delaunay_flip(owner, current, vertex);
    last_added = second_added;
    current = next;
  }
  owner._edges[std::size_t(Owner::opposite(last_added))].boundary = true;
  owner._last_edge = last_added;
  restore_constrained_delaunay(owner);
}

} // namespace tf::topology::cdt

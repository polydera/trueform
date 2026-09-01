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
#include "../../core/buffer.hpp"
#include "./find_constrained_delaunay_edge.hpp"
#include "./queue_constrained_delaunay_flip.hpp"
#include "./reuse_constrained_delaunay_edge.hpp"
#include "./take_deleted_constrained_delaunay_edge.hpp"
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto retriangulate_constrained_delaunay_cavity(
    Owner &owner, const tf::buffer<typename Owner::index_type> &vertices,
    bool clockwise) -> bool {
  using Index = typename Owner::index_type;
  const int required_orientation = clockwise ? 1 : -1;
  owner._retriangulation_stack.clear();
  owner._retriangulation_stack.push_back(vertices[0]);
  owner._retriangulation_stack.push_back(vertices[1]);

  for (Index i = 2; i < static_cast<Index>(vertices.size()); ++i) {
    const Index current = vertices[std::size_t(i)];
    // Popping consumes ears. Fewer than two retained vertices cannot form the
    // next ear and would make the look-behind indices wrap.
    while (owner._retriangulation_stack.size() >= 2) {
      const Index previous_previous =
          owner._retriangulation_stack[owner._retriangulation_stack.size() - 2];
      const Index previous =
          owner._retriangulation_stack[owner._retriangulation_stack.size() - 1];
      if (owner.orient(previous_previous, previous, current) !=
          required_orientation)
        break;

      Index opposite_edge = Owner::none;
      if (clockwise) {
        if (find_constrained_delaunay_edge(owner, previous_previous, current) ==
            Owner::none) {
          opposite_edge = find_constrained_delaunay_edge(
              owner, previous_previous, previous);
          reuse_constrained_delaunay_edge(
              owner, previous_previous, current,
              owner.previous_edge(opposite_edge),
              find_constrained_delaunay_edge(owner, current, previous),
              take_deleted_constrained_delaunay_edge(owner));
        }
      } else if (find_constrained_delaunay_edge(
                     owner, current, previous_previous) == Owner::none) {
        opposite_edge =
            find_constrained_delaunay_edge(owner, previous_previous, previous);
        reuse_constrained_delaunay_edge(
            owner, current, previous_previous,
            owner.previous_edge(
                find_constrained_delaunay_edge(owner, current, previous)),
            opposite_edge, take_deleted_constrained_delaunay_edge(owner));
      }

      owner._retriangulation_stack.pop_back();
      if (i == static_cast<Index>(vertices.size()) - Index(1) &&
          opposite_edge == Owner::none)
        opposite_edge =
            find_constrained_delaunay_edge(owner, previous_previous, previous);
      if (opposite_edge != Owner::none)
        queue_constrained_delaunay_flip(owner, opposite_edge, current);
      if (owner._retriangulation_stack.size() < 2)
        break;
    }
    owner._retriangulation_stack.push_back(current);
  }
  return owner._retriangulation_stack.size() == 2;
}

} // namespace tf::topology::cdt

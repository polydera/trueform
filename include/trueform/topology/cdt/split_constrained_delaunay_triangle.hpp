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
#include "./queue_constrained_delaunay_flip.hpp"
#include "./restore_constrained_delaunay.hpp"

namespace tf::topology::cdt {

/// Insert a vertex strictly inside one face, replacing it with three faces.
/// The original face edges become the edges opposite the new vertex and seed
/// the Delaunay restoration wave.
template <typename Owner>
auto split_constrained_delaunay_triangle(Owner &owner,
                                         typename Owner::index_type vertex,
                                         typename Owner::index_type first)
    -> void {
  using Index = typename Owner::index_type;
  const Index second = owner.previous_edge(Owner::opposite(first));
  const Index third = owner.previous_edge(Owner::opposite(second));
  const Index vertex_a = owner.origin(first);
  const Index vertex_b = owner.origin(second);
  const Index vertex_c = owner.origin(third);

  const Index edge_a = append_constrained_delaunay_edge(
      owner, vertex, vertex_a, Owner::none, first, false);
  const Index edge_b = append_constrained_delaunay_edge(owner, vertex, vertex_b,
                                                        edge_a, second, false);
  append_constrained_delaunay_edge(owner, vertex, vertex_c, edge_b, third,
                                   false);

  queue_constrained_delaunay_flip(owner, first, vertex);
  queue_constrained_delaunay_flip(owner, second, vertex);
  queue_constrained_delaunay_flip(owner, third, vertex);
  owner._locate_hint = edge_a;
  restore_constrained_delaunay(owner);
}

} // namespace tf::topology::cdt

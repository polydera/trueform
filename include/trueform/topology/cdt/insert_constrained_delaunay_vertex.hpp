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
#include "./insert_exterior_constrained_delaunay_vertex.hpp"
#include "./locate_constrained_delaunay.hpp"
#include "./split_constrained_delaunay_boundary_edge.hpp"
#include "./split_constrained_delaunay_triangle.hpp"

namespace tf::topology::cdt {

template <typename Owner>
auto insert_constrained_delaunay_vertex(Owner &owner,
                                        typename Owner::index_type vertex,
                                        typename Owner::locate_result location)
    -> void {
  using LocateKind = typename Owner::locate_kind;
  switch (location.kind) {
  case LocateKind::interior:
    split_constrained_delaunay_triangle(owner, vertex, location.he);
    break;
  case LocateKind::boundary_edge:
    split_constrained_delaunay_boundary_edge(owner, vertex, location.he);
    break;
  case LocateKind::exterior:
    // The exterior fan walk expects the interior twin of the crossed hull edge.
    owner._last_edge = Owner::opposite(location.he);
    insert_exterior_constrained_delaunay_vertex(owner, vertex);
    // Keep the next point-location walk near the vertex just inserted.
    owner._locate_hint = owner._last_edge;
    break;
  }
}

template <typename Owner>
auto insert_constrained_delaunay_vertex(Owner &owner,
                                        typename Owner::index_type vertex)
    -> void {
  insert_constrained_delaunay_vertex(
      owner, vertex, locate_constrained_delaunay(owner, vertex));
}

} // namespace tf::topology::cdt

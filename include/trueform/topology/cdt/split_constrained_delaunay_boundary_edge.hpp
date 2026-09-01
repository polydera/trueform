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
#include "./reuse_constrained_delaunay_edge.hpp"
#include "./unlink_constrained_delaunay_edge.hpp"
#include <cstddef>

namespace tf::topology::cdt {

/// Split a hull edge and its one adjacent interior face. The new vertex ring is
/// ordered toward the first hull endpoint, the opposite face vertex, and the
/// second hull endpoint; the face-next relation depends on that order.
template <typename Owner>
auto split_constrained_delaunay_boundary_edge(
    Owner &owner, typename Owner::index_type vertex,
    typename Owner::index_type boundary_edge) -> void {
  using Index = typename Owner::index_type;
  const Index interior = Owner::opposite(boundary_edge);
  const Index first_vertex = owner.origin(boundary_edge);
  const Index second_vertex = owner.origin(interior);
  const Index first_side = owner.previous_edge(boundary_edge);
  const Index second_side = owner.previous_edge(Owner::opposite(first_side));
  const Index third_vertex = owner.target(first_side);
  unlink_constrained_delaunay_edge(owner, boundary_edge);

  const Index first_to_new = reuse_constrained_delaunay_edge(
      owner, first_vertex, vertex, first_side, Owner::none, boundary_edge);
  const Index new_to_first = Owner::opposite(first_to_new);
  owner._edges[std::size_t(first_to_new)].boundary = true;
  owner._edges[std::size_t(first_to_new)].delaunay = true;

  const Index new_to_third = append_constrained_delaunay_edge(
      owner, vertex, third_vertex, new_to_first, second_side, false);
  const Index new_to_second = append_constrained_delaunay_edge(
      owner, vertex, second_vertex, new_to_third,
      owner.previous_edge(Owner::opposite(second_side)), false);
  owner._edges[std::size_t(new_to_second)].boundary = true;
  owner._edges[std::size_t(new_to_second)].delaunay = true;

  queue_constrained_delaunay_flip(owner, first_side, vertex);
  queue_constrained_delaunay_flip(owner, second_side, vertex);
  owner._last_edge = first_to_new;
  owner._locate_hint = new_to_third;
  restore_constrained_delaunay(owner);
}

} // namespace tf::topology::cdt

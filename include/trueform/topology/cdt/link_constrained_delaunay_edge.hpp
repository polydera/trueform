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
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto link_constrained_delaunay_edge(Owner &owner,
                                    typename Owner::index_type new_edge,
                                    typename Owner::index_type vertex,
                                    typename Owner::index_type after) -> void {
  owner._edges[std::size_t(new_edge)].vertex = vertex;
  if (after == Owner::none) {
    owner._v_first_edge[std::size_t(vertex)] = new_edge;
    owner._edges[std::size_t(new_edge)].prev = new_edge;
    owner._edges[std::size_t(new_edge)].next = new_edge;
    return;
  }

  const auto before = owner.next_edge(after);
  owner._edges[std::size_t(new_edge)].prev = after;
  owner._edges[std::size_t(new_edge)].next = before;
  owner._edges[std::size_t(after)].next = new_edge;
  owner._edges[std::size_t(before)].prev = new_edge;
}

} // namespace tf::topology::cdt

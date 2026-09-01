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
auto unlink_constrained_delaunay_edge(Owner &owner,
                                      typename Owner::index_type edge) -> void {
  const auto opposite = Owner::opposite(edge);
  owner._v_first_edge[std::size_t(owner.origin(edge))] = owner.next_edge(edge);
  owner._v_first_edge[std::size_t(owner.origin(opposite))] =
      owner.next_edge(opposite);

  owner._edges[std::size_t(owner.next_edge(edge))].prev =
      owner.previous_edge(edge);
  owner._edges[std::size_t(owner.previous_edge(edge))].next =
      owner.next_edge(edge);
  owner._edges[std::size_t(owner.next_edge(opposite))].prev =
      owner.previous_edge(opposite);
  owner._edges[std::size_t(owner.previous_edge(opposite))].next =
      owner.next_edge(opposite);
}

} // namespace tf::topology::cdt

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
#include "./link_constrained_delaunay_edge.hpp"
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto append_constrained_delaunay_edge(Owner &owner,
                                      typename Owner::index_type first_vertex,
                                      typename Owner::index_type second_vertex,
                                      typename Owner::index_type after_first,
                                      typename Owner::index_type after_second,
                                      bool boundary) ->
    typename Owner::index_type {
  using Index = typename Owner::index_type;
  const Index edge = static_cast<Index>(owner._edges.size());
  owner._edges.push_back(typename Owner::half_edge{
      Owner::none, Owner::none, Owner::none, boundary, false, false});
  owner._edges.push_back(typename Owner::half_edge{
      Owner::none, Owner::none, Owner::none, boundary, false, false});
  link_constrained_delaunay_edge(owner, edge, first_vertex, after_first);
  link_constrained_delaunay_edge(owner, Owner::opposite(edge), second_vertex,
                                 after_second);
  return edge;
}

} // namespace tf::topology::cdt

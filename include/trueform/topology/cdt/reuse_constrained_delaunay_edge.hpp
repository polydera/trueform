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
auto reuse_constrained_delaunay_edge(Owner &owner,
                                     typename Owner::index_type first_vertex,
                                     typename Owner::index_type second_vertex,
                                     typename Owner::index_type after_first,
                                     typename Owner::index_type after_second,
                                     typename Owner::index_type reused) ->
    typename Owner::index_type {
  owner._edges[std::size_t(reused)] = typename Owner::half_edge{
      Owner::none, Owner::none, Owner::none, false, false, false};
  owner._edges[std::size_t(Owner::opposite(reused))] =
      typename Owner::half_edge{Owner::none, Owner::none, Owner::none,
                                false,       false,       false};
  link_constrained_delaunay_edge(owner, reused, first_vertex, after_first);
  link_constrained_delaunay_edge(owner, Owner::opposite(reused), second_vertex,
                                 after_second);
  return reused;
}

} // namespace tf::topology::cdt

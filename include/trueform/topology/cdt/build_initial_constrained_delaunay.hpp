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
#include "./build_delaunay_topology.hpp"
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto build_initial_constrained_delaunay(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  using ExecutionPolicy = typename Owner::execution_policy;
  const auto result =
      build_delaunay_topology<ExecutionPolicy>(owner, Owner::none);

  Index outer = Owner::opposite(result.outer_edge);
  const Index outer_seed = outer;
  do {
    owner._edges[std::size_t(outer)].boundary = true;
    outer = owner.previous_edge(Owner::opposite(outer));
  } while (outer != outer_seed);
  owner._last_edge = Owner::opposite(outer_seed);
}

} // namespace tf::topology::cdt

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
#include "./abandon_constrained_delaunay.hpp"
#include "./build_initial_constrained_delaunay.hpp"
#include "./rebuild_constrained_delaunay_vertex_edges.hpp"
#include "./recover_constrained_delaunay_edge.hpp"
#include "./remap_constrained_delaunay_vertices.hpp"
#include "./restore_constrained_delaunay.hpp"
#include <cstddef>

namespace tf::topology::cdt {

/// Build the retained topology from already converted, welded points and
/// prepared constraint identities.
template <typename Owner>
auto build_prepared_constrained_delaunay(Owner &owner) -> bool {
  using Index = typename Owner::index_type;
  using Param = typename Owner::param_type;
  const std::size_t n_vertices = owner._points.size();
  if (n_vertices < 3)
    return true;

  owner._edges.reserve(9 * n_vertices);
  if (owner._final_constraint_edges.size() != 0)
    owner._flip_stack.reserve(6 * n_vertices);

  build_initial_constrained_delaunay(owner);
  if (owner._final_constraint_edges.size() != 0) {
    remap_constrained_delaunay_vertices(owner);
    rebuild_constrained_delaunay_vertex_edges(owner, n_vertices);
  }

  const Param maximum = Param(1) << Owner::crossing_param_bits;
  for (Index i = 0;
       i < static_cast<Index>(owner._final_constraint_edges.size()); ++i) {
    const auto edge = owner._final_constraint_edges[std::size_t(i)];
    const Index first = edge[0];
    const Index second = edge[1];
    if (first == second)
      continue;
    const Index input_id =
        i < static_cast<Index>(owner._constraint_input_ids.size())
            ? owner._constraint_input_ids[std::size_t(i)]
            : Owner::none;
    if (!recover_constrained_delaunay_edge(
            owner, first, second, owner._final_is_boundary[std::size_t(i)] != 0,
            input_id, Param(0), maximum, 0, false)) {
      abandon_constrained_delaunay(owner);
      return false;
    }
  }

  if (owner._final_constraint_edges.size() != 0)
    restore_constrained_delaunay(owner);
  return true;
}

} // namespace tf::topology::cdt

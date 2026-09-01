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
#include "./constrained_delaunay_edge_constraint_provenance.hpp"
#include "./constrained_delaunay_face_edges.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto constrained_delaunay_face_constraint_owners(
    const Owner &owner, typename Owner::index_type triangle)
    -> std::array<typename Owner::constraint_provenance, 3> {
  using Index = typename Owner::index_type;
  using Param = typename Owner::param_type;
  using Provenance = typename Owner::constraint_provenance;
  const auto edges = constrained_delaunay_face_edges(owner, triangle);
  const auto provenance = [&](Index edge) -> Provenance {
    if (owner._edges[std::size_t(edge)].constrained == Owner::unconstrained)
      return {Owner::none, Param(0), Param(0)};
    return constrained_delaunay_edge_constraint_provenance(owner, edge);
  };
  return {provenance(edges[0]), provenance(edges[1]), provenance(edges[2])};
}

} // namespace tf::topology::cdt

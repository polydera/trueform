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
#include "./constrained_delaunay_face_edges.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto constrained_delaunay_face_constraints(const Owner &owner,
                                           typename Owner::index_type triangle)
    -> std::array<typename Owner::index_type, 3> {
  using Index = typename Owner::index_type;
  const auto edges = constrained_delaunay_face_edges(owner, triangle);
  const auto input_id = [&](Index edge) {
    if (owner._edges[std::size_t(edge)].constrained == Owner::unconstrained)
      return Owner::none;
    return owner._constraint_provenance.get(edge, Owner::none).input_id;
  };
  return {input_id(edges[0]), input_id(edges[1]), input_id(edges[2])};
}

} // namespace tf::topology::cdt

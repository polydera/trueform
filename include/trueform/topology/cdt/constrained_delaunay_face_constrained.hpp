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
auto constrained_delaunay_face_constrained(const Owner &owner,
                                           typename Owner::index_type triangle)
    -> std::array<bool, 3> {
  const auto edges = constrained_delaunay_face_edges(owner, triangle);
  return {
      owner._edges[std::size_t(edges[0])].constrained != Owner::unconstrained,
      owner._edges[std::size_t(edges[1])].constrained != Owner::unconstrained,
      owner._edges[std::size_t(edges[2])].constrained != Owner::unconstrained};
}

} // namespace tf::topology::cdt

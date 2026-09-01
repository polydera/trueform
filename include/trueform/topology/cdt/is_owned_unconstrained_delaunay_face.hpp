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
#include "./unconstrained_delaunay_triangle_orbit.hpp"
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner, typename VisitBuffer>
auto is_owned_unconstrained_delaunay_face(
    const Owner &owner, const VisitBuffer &visited,
    const unconstrained_delaunay_triangle_orbit<typename Owner::index_type>
        &orbit) -> bool {
  const auto first = orbit.darts.front();
  if (!orbit.closed || visited[std::size_t(first)] != 0 ||
      owner._edges[std::size_t(first)].vertex == Owner::none)
    return false;
  return first < orbit.darts[1] && first < orbit.darts[2];
}

} // namespace tf::topology::cdt

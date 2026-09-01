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
#include <cstdint>

namespace tf::topology::cdt {

template <typename VisitBuffer, typename Index>
auto mark_unconstrained_delaunay_face_visited(
    VisitBuffer &visited,
    const unconstrained_delaunay_triangle_orbit<Index> &orbit) -> void {
  for (const Index dart : orbit.darts)
    visited[std::size_t(dart)] = std::uint8_t(1);
}

} // namespace tf::topology::cdt

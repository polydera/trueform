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
#include "./for_each_unconstrained_delaunay_face_dart.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt {

template <typename Owner, typename VisitBuffer>
auto mark_unconstrained_delaunay_outer_face(
    const Owner &owner, typename Owner::index_type outer_seed,
    VisitBuffer &visited) -> void {
  for_each_unconstrained_delaunay_face_dart(
      owner, outer_seed, [&](typename Owner::index_type dart) {
        visited[std::size_t(dart)] = std::uint8_t(1);
      });
}

} // namespace tf::topology::cdt

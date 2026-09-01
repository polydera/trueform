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
#include "./mark_unconstrained_delaunay_outer_face.hpp"
#include "./materialize_unconstrained_delaunay_faces_parallel.hpp"
#include "./materialize_unconstrained_delaunay_faces_serial.hpp"
#include <cstdint>

namespace tf::topology::cdt {

template <typename ExecutionPolicy, typename Owner>
auto materialize_unconstrained_delaunay_faces(
    Owner &owner, typename Owner::index_type outer_seed) -> void {
  auto &visited = owner._face_visited;
  visited.allocate_and_initialize(owner._edges.size(), std::uint8_t(0));
  mark_unconstrained_delaunay_outer_face(owner, outer_seed, visited);
  if constexpr (ExecutionPolicy::parallel)
    materialize_unconstrained_delaunay_faces_parallel(owner, visited);
  else
    materialize_unconstrained_delaunay_faces_serial(owner, visited);
}

} // namespace tf::topology::cdt

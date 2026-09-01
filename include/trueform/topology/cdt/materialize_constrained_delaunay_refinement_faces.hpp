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
#include "../../core/blocked_buffer.hpp"

namespace tf::topology::cdt {

template <typename Owner>
auto materialize_constrained_delaunay_refinement_faces(const Owner &owner)
    -> tf::blocked_buffer<typename Owner::index_type, 3> {
  using Index = typename Owner::index_type;

  tf::blocked_buffer<Index, 3> out;
  out.reserve(owner._t.size());
  for (const auto &face : owner._t)
    out.push_back({face.v[0], face.v[1], face.v[2]});
  return out;
}

} // namespace tf::topology::cdt

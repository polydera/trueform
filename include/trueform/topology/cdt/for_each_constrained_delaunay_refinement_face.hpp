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
#include <array>
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner, typename F>
auto for_each_constrained_delaunay_refinement_face(const Owner &owner,
                                                   F &&function) -> void {
  using Index = typename Owner::index_type;

  for (Index triangle = 0; triangle < static_cast<Index>(owner._t.size());
       ++triangle) {
    const auto &face = owner._t[std::size_t(triangle)];
    function(triangle, face.v[0], face.v[1], face.v[2],
             owner._label[std::size_t(triangle)],
             std::array<Index, 3>{face.n[0], face.n[1], face.n[2]},
             std::array<bool, 3>{face.seg[0] != Owner::none,
                                 face.seg[1] != Owner::none,
                                 face.seg[2] != Owner::none});
  }
}

} // namespace tf::topology::cdt

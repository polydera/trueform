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
#include "./unconstrained_delaunay_face_next.hpp"

namespace tf::topology::cdt {

/// Visit one complete left-face orbit, beginning with `first`.
template <typename Owner, typename Function>
auto for_each_unconstrained_delaunay_face_dart(const Owner &owner,
                                               typename Owner::index_type first,
                                               Function &&function) -> void {
  auto current = first;
  for (;;) {
    function(current);
    current = unconstrained_delaunay_face_next(owner, current);
    if (current == first)
      return;
  }
}

} // namespace tf::topology::cdt

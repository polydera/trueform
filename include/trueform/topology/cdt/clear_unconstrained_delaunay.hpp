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

namespace tf::topology::cdt {

template <typename Owner>
auto clear_unconstrained_delaunay(Owner &owner) -> void {
  owner.clear_delaunay_topology();
  owner._faces.clear();
  owner._face_offsets.clear();
  owner._face_visited.clear();
}

} // namespace tf::topology::cdt

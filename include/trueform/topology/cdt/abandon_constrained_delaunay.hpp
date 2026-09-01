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
#include "./clear_constrained_delaunay_products.hpp"

namespace tf::topology::cdt {

template <typename Owner>
auto abandon_constrained_delaunay(Owner &owner) -> void {
  owner._edges.clear();
  owner._sites.clear();
  owner._keys.clear();
  clear_constrained_delaunay_products(owner);
  owner._scratch_face_offsets.clear();
  owner._scratch_stack.clear();
}

} // namespace tf::topology::cdt

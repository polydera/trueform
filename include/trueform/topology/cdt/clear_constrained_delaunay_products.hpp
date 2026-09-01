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
#include "../cdt_region_mode.hpp"

namespace tf::topology::cdt {

template <typename Owner>
auto clear_constrained_delaunay_products(Owner &owner) -> void {
  owner._first_edge_of_triangle.clear();
  owner._triangle_of_edge.clear();
  owner._faces.clear();
  owner._region_labels.clear();
  owner._region_mode = tf::cdt_region_mode::nesting;
}

} // namespace tf::topology::cdt

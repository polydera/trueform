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
#include "./build_prepared_constrained_delaunay.hpp"
#include "./clear_constrained_delaunay_products.hpp"
#include "./materialize_constrained_delaunay_faces.hpp"

namespace tf::topology::cdt {

/// Recover prepared constraints and emit triangle vertices without building
/// adjacency or region labels.
template <typename Owner>
auto build_constrained_delaunay_from_prepared_constraints(Owner &owner)
    -> bool {
  clear_constrained_delaunay_products(owner);
  if (!build_prepared_constrained_delaunay(owner))
    return false;
  materialize_constrained_delaunay_faces(owner);
  return true;
}

} // namespace tf::topology::cdt

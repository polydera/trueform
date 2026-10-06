/*
 * Copyright (c) 2026 XLAB
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

namespace tf::topology::cdt::dt3 {

/// Clear cell products, retaining prepared sites and their input map.
template <typename Owner>
auto clear_tetrahedralization_cells(Owner &owner) -> void {
  using Index = typename Owner::index_type;
  owner._corners.clear();
  owner._neighbors.clear();
  owner._states.clear();
  owner._cavity.clear();
  owner._boundary.clear();
  owner._sides.clear();
  owner._hint = Index(0);
  owner._n_dead = 0;
  owner._n_finite = 0;
}

} // namespace tf::topology::cdt::dt3

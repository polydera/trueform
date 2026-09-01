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
auto constrained_delaunay_edge_constraint_provenance(
    const Owner &owner, typename Owner::index_type edge) ->
    typename Owner::constraint_provenance {
  return owner._constraint_provenance.get(edge, Owner::none);
}

} // namespace tf::topology::cdt

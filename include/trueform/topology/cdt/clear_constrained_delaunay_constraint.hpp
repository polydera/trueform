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
#include <cstddef>

namespace tf::topology::cdt {

template <typename Owner>
auto clear_constrained_delaunay_constraint(Owner &owner,
                                           typename Owner::index_type edge)
    -> void {
  owner._edges[std::size_t(edge)].constrained = Owner::unconstrained;
  owner._edges[std::size_t(Owner::opposite(edge))].constrained =
      Owner::unconstrained;
}

} // namespace tf::topology::cdt

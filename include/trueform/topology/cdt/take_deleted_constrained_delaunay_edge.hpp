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
auto take_deleted_constrained_delaunay_edge(Owner &owner) ->
    typename Owner::index_type {
  const auto edge = owner._deleted_edges.back();
  owner._deleted_edges.pop_back();
  return edge;
}

} // namespace tf::topology::cdt

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
auto unconstrained_delaunay_face_next(const Owner &owner,
                                      typename Owner::index_type edge) ->
    typename Owner::index_type {
  using Index = typename Owner::index_type;
  const Index opposite = edge ^ Index(1);
  return owner._edges[std::size_t(opposite)].prev;
}

} // namespace tf::topology::cdt

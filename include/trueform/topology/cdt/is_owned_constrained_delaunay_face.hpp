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
auto is_owned_constrained_delaunay_face(const Owner &owner,
                                        typename Owner::index_type first,
                                        typename Owner::index_type &second,
                                        typename Owner::index_type &third)
    -> bool {
  if (owner._edges[std::size_t(first)].boundary)
    return false;
  second = owner.previous_edge(Owner::opposite(first));
  if (second == Owner::none)
    return false;
  third = owner.previous_edge(Owner::opposite(second));
  return third != Owner::none && !owner._edges[std::size_t(second)].boundary &&
         !owner._edges[std::size_t(third)].boundary && first < second &&
         first < third;
}

} // namespace tf::topology::cdt

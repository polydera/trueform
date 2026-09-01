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
#include "../../core/point.hpp"
#include "../../exact/is_between_on_segment.hpp"

namespace tf::topology::cdt {

template <typename Owner>
auto is_between_constrained_delaunay_vertices(const Owner &owner,
                                              typename Owner::index_type first,
                                              typename Owner::index_type second,
                                              typename Owner::index_type query)
    -> bool {
  using Int = typename Owner::int_type;
  return tf::exact::is_between_on_segment<Int>(
      owner.point(first), owner.point(second), owner.point(query));
}

template <typename Owner>
auto is_between_constrained_delaunay_vertices(
    const Owner &owner, typename Owner::index_type first,
    typename Owner::index_type second,
    const tf::point<typename Owner::int_type, 2> &query) -> bool {
  using Int = typename Owner::int_type;
  return tf::exact::is_between_on_segment<Int>(owner.point(first),
                                               owner.point(second), query);
}

} // namespace tf::topology::cdt

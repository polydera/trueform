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
#include "../../exact/dyadic_blend.hpp"

#include "../../core/point.hpp"
#include "../../exact/meta.hpp"
#include "../detail/region_triangulation_types.hpp"

namespace tf::cut {

template <typename Int>
auto region_split_point(const tf::point<Int, 3> &a,
                        const tf::point<Int, 3> &b,
                        typename tf::exact::meta<Int>::param_type parameter,
                        int axis_0, int axis_1) -> tf::point<Int, 2> {
  return {tf::exact::dyadic_blend(a[axis_0], b[axis_0], parameter,
                                  tf::exact::meta<Int>::split_grid_bits),
          tf::exact::dyadic_blend(a[axis_1], b[axis_1], parameter,
                                  tf::exact::meta<Int>::split_grid_bits)};
}

} // namespace tf::cut

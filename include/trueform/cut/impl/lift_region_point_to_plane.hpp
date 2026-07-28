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

#include "../../core/point.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/orient3d.hpp"
#include "../detail/region_triangulation_types.hpp"
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int>
auto lift_region_point_to_plane(
    const tf::cut::detail::region_triangulation_workspace<Index, Int>
        &workspace,
    const tf::point<Int, 2> &point) -> tf::point<Int, 3> {
  tf::point<Int, 3> lifted;
  const int axis_0 = workspace.ax0;
  const int axis_1 = workspace.ax1;
  const int axis_2 = 3 - axis_0 - axis_1;
  lifted[axis_0] = point[0];
  lifted[axis_1] = point[1];
  const auto numerator =
      workspace.plane_d -
      workspace.plane_n[std::size_t(axis_0)] *
          typename tf::exact::meta<Int>::T2(point[0]) -
      workspace.plane_n[std::size_t(axis_1)] *
          typename tf::exact::meta<Int>::T2(point[1]);
  const auto denominator =
      workspace.plane_n[std::size_t(axis_2)];
  lifted[axis_2] =
      denominator == typename tf::exact::meta<Int>::T2(0)
          ? workspace.plane_fallback
          : static_cast<Int>(
                tf::exact::div_round(numerator, denominator));
  return lifted;
}

} // namespace tf::cut

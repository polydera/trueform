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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"
#include "../../exact/wide_dyadic_ratio.hpp"
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// ONE IDENTITY, A PARAMETER PER SIDE: a point the drain states on a side
/// `(a, b)` — a row `{a, b, point}` of `edge_points`, in slots — stands
/// there at the dyadic parameter `t[r]` of its projection from `a`, its own
/// identity and position untouched; the chart reads it there
/// (@ref state_tet_lane_occurrences). `slot_point(slot)` reads a side's end
/// and `point(id)` a point's position. O(rows).
template <typename Int, typename Index, typename SlotPoint, typename Point>
auto state_tet_side_parameters(
    const tf::buffer<std::array<Index, 3>> &edge_points,
    const SlotPoint &slot_point, const Point &point,
    tf::buffer<typename tf::exact::meta<Int>::param_type> &t) -> void {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  const std::size_t n = edge_points.size();
  t.allocate(n);
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&](std::size_t r) {
        const auto &row = edge_points[r];
        const tf::exact::pt3<Int> a = slot_point(row[0]),
                                  b = slot_point(row[1]);
        const tf::exact::pt3<Int> p = point(row[2]);
        T2 length2(0), along(0);
        for (std::size_t d = 0; d < 3; ++d) {
          const T1 side = T1(b[d]) - T1(a[d]);
          length2 = length2 + T2(side) * T2(side);
          along = along + T2(side) * T2(T1(p[d]) - T1(a[d]));
        }
        t[r] = tf::exact::wide_dyadic_ratio<Int>(along, length2);
      },
      tf::checked);
}

} // namespace tf::tetmesh

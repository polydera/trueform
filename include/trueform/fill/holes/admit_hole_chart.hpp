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
#include "../../core/cross.hpp"
#include "../../core/dot.hpp"
#include "../../core/polygons.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/orient2d.hpp"
#include "../hole_fill_status.hpp"
#include "./hole_chart.hpp"
#include "./hole_collar.hpp"
#include <cmath>
#include <cstddef>

namespace tf::fill {

/// Admit one chart, or state why it does not hold its hole.
///
/// The rim's area and every ring-1 fold are decided exactly on the chart's
/// lattice; the slope is a policy read in floating point, asked of a collar
/// that does not fold.
template <typename Policy, typename Index, typename Int>
auto admit_hole_chart(const tf::polygons<Policy> &polygons,
                      const tf::fill::hole_collar<Index> &collar,
                      const tf::fill::hole_chart<Int> &chart)
    -> tf::hole_chart_status {
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto points = polygons.points();
  const std::size_t n = collar.n_rim();

  T2 area = T2(0);
  for (std::size_t k = 1; k + 1 < n; ++k)
    area = area + tf::exact::orient2d(chart.lattice[0], chart.lattice[k],
                                      chart.lattice[k + 1]);
  if (!(area < T2(0)))
    return tf::hole_chart_status::no_area;

  for (const auto &triangle : collar.ring_triangles)
    if (tf::exact::orient2d_sign(chart.lattice[std::size_t(triangle[0])],
                                 chart.lattice[std::size_t(triangle[1])],
                                 chart.lattice[std::size_t(triangle[2])]) <= 0)
      return tf::hole_chart_status::folded;

  for (std::size_t f = 0; f < collar.ring_triangles.size(); ++f) {
    const auto corners = polygons.faces()[std::size_t(collar.faces[f])];
    const auto p0 = points[std::size_t(corners[0])].template as<double>();
    const auto p1 = points[std::size_t(corners[1])].template as<double>();
    const auto p2 = points[std::size_t(corners[2])].template as<double>();
    const auto normal = tf::cross(p1 - p0, p2 - p0);
    const double length = std::sqrt(tf::dot(normal, normal));
    if (!(tf::dot(normal, chart.normal) >=
          tf::fill::hole_chart_slope_cosine * length))
      return tf::hole_chart_status::steep;
  }
  return tf::hole_chart_status::admitted;
}

} // namespace tf::fill

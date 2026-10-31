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
#include "../../core/buffer.hpp"
#include "./hole_chart.hpp"
#include "./hole_collar.hpp"
#include "./hole_trend.hpp"
#include <cstddef>

namespace tf::fill {

/// The residual the trend leaves at every rim and ring-1 position, in the
/// chart's own unit, the rim's radius — the values a residual solve holds
/// fixed, in collar order. Returns false where the trend does not cover one
/// of them.
template <typename Index, typename Int>
auto state_hole_chart_fixed_residuals(
    const tf::fill::hole_collar<Index> &collar,
    const tf::fill::hole_chart<Int> &chart, const tf::fill::hole_trend &trend,
    tf::buffer<double> &values) -> bool {
  const std::size_t fixed = std::size_t(collar.vertex_offsets[2]);
  const double scale = 1.0 / chart.radius;
  values.allocate(fixed);
  for (std::size_t position = 0; position < fixed; ++position) {
    double trended = 0.0;
    if (!tf::fill::hole_trend_height(trend, chart, chart.uv[position], trended))
      return false;
    values[position] = (chart.heights[position] - trended) * scale;
  }
  return true;
}

} // namespace tf::fill

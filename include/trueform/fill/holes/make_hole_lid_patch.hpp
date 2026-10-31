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
#include "../../core/polygons.hpp"
#include "../hole_fill_status.hpp"
#include "./admit_hole_chart.hpp"
#include "./hole_chart.hpp"
#include "./hole_trend.hpp"
#include "./make_hole_chart_patch.hpp"
#include "./make_hole_rim_collar.hpp"
#include "./place_hole_chart_samples.hpp"
#include "./sample_hole_chart.hpp"

namespace tf::fill {

/// Lay the flat lid of a rim that lies exactly in one plane, the rim given as
/// host vertex ids in its own order.
///
/// The chart stands on the rim alone, so no ring of the host enters it and
/// nothing the host states can refuse it: the trend is the chart's own plane
/// and the residual is zero. The lid splits no rim edge. The product is the
/// one @ref tf::fill::make_hole_chart_patch states, positional over
/// `[rim | samples]`.
template <typename Policy, typename RimRange, typename Index, typename Int,
          typename RealT>
auto make_hole_lid_patch(
    const tf::polygons<Policy> &polygons, const RimRange &rim, double unit,
    tf::fill::hole_chart_scratch<Index, Int, RealT> &scratch,
    tf::hole_refine_status &refined) -> tf::hole_chart_status {
  tf::fill::make_hole_rim_collar(rim, scratch.collar);
  const auto charted =
      tf::fill::make_hole_chart(polygons, scratch.collar, unit, scratch.chart);
  if (charted != tf::hole_chart_status::admitted)
    return charted;
  const auto verdict =
      tf::fill::admit_hole_chart(polygons, scratch.collar, scratch.chart);
  if (verdict != tf::hole_chart_status::admitted)
    return verdict;
  scratch.trend = {tf::fill::hole_trend_kind::quadratic, {}};
  if (!tf::fill::sample_hole_chart(scratch.collar, scratch.chart, false,
                                   scratch.sample, scratch.samples,
                                   scratch.triangles, scratch.rim_splits))
    return tf::hole_chart_status::unsampled;
  scratch.residual.allocate_and_initialize(scratch.samples.size(), 0.0);
  if (!tf::fill::place_hole_chart_samples(scratch.chart, scratch.trend,
                                          scratch.samples, scratch.residual,
                                          scratch.points))
    return tf::hole_chart_status::uncovered;
  refined = tf::fill::hole_chart_refine_status(scratch.sample);
  return tf::hole_chart_status::rim_plane;
}

} // namespace tf::fill

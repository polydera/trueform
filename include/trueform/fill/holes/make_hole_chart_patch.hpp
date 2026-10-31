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
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../topology/face_membership_like.hpp"
#include "../hole_fill_status.hpp"
#include "./admit_hole_chart.hpp"
#include "./fit_hole_chart_residual_basis.hpp"
#include "./extend_hole_chart_rim.hpp"
#include "./fit_hole_trend.hpp"
#include "./hole_chart.hpp"
#include "./hole_collar.hpp"
#include "./hole_trend.hpp"
#include "./place_hole_chart_samples.hpp"
#include "./sample_hole_chart.hpp"
#include "./solve_hole_chart_residual.hpp"
#include "./state_hole_chart_fixed_residuals.hpp"
#include <array>

namespace tf::fill {

/// Which solve answered a chart's residual.
enum class hole_chart_solver { mesh, basis };

/// The scratch one chart patch is made on, reused across the rims a worker
/// takes. `points`, `triangles` and `rim_splits` are the product: the placed
/// samples, the triangles over `[extended rim | samples]` positions, and the
/// splits the rim was extended by; `solver` states which solve answered the
/// residual they stand on.
template <typename Index, typename Int, typename RealT>
struct hole_chart_scratch {
  tf::fill::hole_collar_scratch<Index> gather;
  tf::fill::hole_collar<Index> collar;
  tf::fill::hole_chart<Int> chart;
  tf::fill::hole_trend_scratch fit;
  tf::fill::hole_trend trend;
  tf::fill::hole_chart_sample_scratch<Index, Int> sample;
  tf::buffer<tf::point<double, 2>> samples;
  tf::buffer<tf::fill::hole_chart_rim_split<Index, Int>> rim_splits;
  tf::fill::hole_chart_extend_scratch<Index, Int> extend;
  tf::buffer<double> fixed;
  tf::fill::hole_chart_basis_scratch basis;
  tf::fill::hole_chart_solve_scratch<Index> solve;
  tf::fill::hole_chart_solver solver;
  tf::buffer<double> residual;
  tf::buffer<tf::point<RealT, 3>> points;
  tf::buffer<std::array<Index, 3>> triangles;
};

/// Read one rim's hole as a chart and stand its samples, the rim given as
/// host vertex ids in its own order.
///
/// The product is positional: `triangles` name `[extended rim | samples]`
/// positions, each wound positively in the chart — against the rim, which
/// runs the host's way — and `points` holds sample `j` at the position after
/// the extended rim's last. Where `split_rim` allows it the sampling may
/// split rim edges, and the collar and chart are extended by those splits
/// (@ref tf::fill::extend_hole_chart_rim) before anything reads them again.
/// The rim itself is never moved. `refined` states how the sampling
/// refinement left the chart. Any refusal leaves the product meaningless and
/// names the fact that refused it.
///
/// A chart of at least @ref tf::fill::hole_chart_basis_samples samples first
/// offers its residual to @ref tf::fill::fit_hole_chart_residual_basis; where
/// the basis's certificate does not hold, the mesh solve answers, as it does
/// for every smaller chart.
template <typename Policy, typename MembershipPolicy, typename RimRange,
          typename Index, typename Int, typename RealT>
auto make_hole_chart_patch(
    const tf::polygons<Policy> &polygons,
    const tf::face_membership_like<MembershipPolicy> &fm, const RimRange &rim,
    double unit, bool split_rim,
    tf::fill::hole_chart_scratch<Index, Int, RealT> &scratch,
    tf::hole_refine_status &refined) -> tf::hole_chart_status {
  if (!tf::fill::gather_hole_collar(polygons, fm, rim, scratch.gather,
                                    scratch.collar))
    return tf::hole_chart_status::collar_arity;
  const auto charted =
      tf::fill::make_hole_chart(polygons, scratch.collar, unit, scratch.chart);
  if (charted != tf::hole_chart_status::admitted)
    return charted;
  const auto verdict =
      tf::fill::admit_hole_chart(polygons, scratch.collar, scratch.chart);
  if (verdict != tf::hole_chart_status::admitted)
    return verdict;
  if (!tf::fill::fit_hole_trend(polygons, scratch.collar, scratch.chart,
                                scratch.fit, scratch.trend))
    return tf::hole_chart_status::uncovered;
  if (!tf::fill::sample_hole_chart(scratch.collar, scratch.chart, split_rim,
                                   scratch.sample, scratch.samples,
                                   scratch.triangles, scratch.rim_splits))
    return tf::hole_chart_status::unsampled;
  tf::fill::extend_hole_chart_rim(polygons, rim, scratch.rim_splits,
                                  scratch.extend, scratch.collar,
                                  scratch.chart);
  scratch.solver = tf::fill::hole_chart_solver::mesh;
  scratch.residual.clear();
  if (scratch.samples.size()) {
    if (!tf::fill::state_hole_chart_fixed_residuals(
            scratch.collar, scratch.chart, scratch.trend, scratch.fixed))
      return tf::hole_chart_status::uncovered;
    if (scratch.samples.size() >= tf::fill::hole_chart_basis_samples &&
        tf::fill::fit_hole_chart_residual_basis(
            scratch.collar, scratch.chart, scratch.fixed, scratch.samples,
            scratch.basis, scratch.residual))
      scratch.solver = tf::fill::hole_chart_solver::basis;
    else if (!tf::fill::solve_hole_chart_residual(
                 scratch.collar, scratch.chart, scratch.fixed, scratch.samples,
                 scratch.triangles, scratch.solve, scratch.residual))
      return tf::hole_chart_status::unsolved;
  }
  if (!tf::fill::place_hole_chart_samples(scratch.chart, scratch.trend,
                                          scratch.samples, scratch.residual,
                                          scratch.points))
    return tf::hole_chart_status::uncovered;
  refined = tf::fill::hole_chart_refine_status(scratch.sample);
  return tf::hole_chart_status::admitted;
}

} // namespace tf::fill

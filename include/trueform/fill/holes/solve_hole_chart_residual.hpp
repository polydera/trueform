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
#include "../../geometry/dirichlet_field_status.hpp"
#include "../../geometry/laplacian/assemble_laplacian_system.hpp"
#include "../../geometry/laplacian/laplacian_form.hpp"
#include "../../geometry/laplacian/laplacian_is_anchored.hpp"
#include "../../geometry/laplacian/laplacian_system.hpp"
#include "../../geometry/laplacian/make_laplacian_form_pattern.hpp"
#include "../../geometry/laplacian/solve_dirichlet_form.hpp"
#include "../../geometry/laplacian/state_dirichlet_load.hpp"
#include "../../geometry/laplacian/state_laplacian_form_values.hpp"
#include "../../topology/face_membership.hpp"
#include "../../topology/policy/face_membership.hpp"
#include "./hole_chart.hpp"
#include "./hole_collar.hpp"
#include <array>
#include <cstddef>

namespace tf::fill {

/// The scratch one residual solve walks on. The operator and its solve are
/// the fairing's own, read here on one scalar over the chart.
template <typename Index> struct hole_chart_solve_scratch {
  tf::face_membership<Index> membership;
  tf::geometry::laplacian_assembly_scratch assembly;
  tf::geometry::laplacian_system<Index> system;
  tf::geometry::laplacian_form<Index> form;
  tf::geometry::dirichlet_solve_scratch<Index> solve;
  tf::buffer<tf::point<double, 3>> positions;
  tf::buffer<double> values;
  tf::buffer<double> load;
  tf::buffer<std::array<Index, 3>> triangles;
  tf::buffer<Index> free_ids;
  tf::buffer<Index> free_of;
  tf::buffer<Index> stack;
  tf::buffer<char> seen;
};

/// Solve the residual the trend leaves at the samples.
///
/// The solve runs in the chart's own unit, the rim's radius, heights as
/// well as positions, so its tolerance means the same thing at any scale of
/// the model.
///
/// The operator is the cotangent Laplacian and lumped mass of the chart:
/// `triangles` over `[rim | samples]` positions, each wound with the chart,
/// joined to the collar's ring-1 faces. The rim and ring 1 hold `fixed`,
/// the residual the trend leaves at their own heights
/// (@ref tf::fill::state_hole_chart_fixed_residuals), and the samples take
/// the discrete biharmonic answer, so the residual is read at positions only
/// — a host slope never enters it. Every sample must be a corner of a
/// triangle. Returns false where the operator states a degenerate triangle,
/// a run of samples reaches no held row, or the operator is not definite.
///
/// `residual` receives one height per sample, in `samples` order.
template <typename Index, typename Int, typename Samples, typename Triangles>
auto solve_hole_chart_residual(
    const tf::fill::hole_collar<Index> &collar,
    const tf::fill::hole_chart<Int> &chart, const tf::buffer<double> &fixed,
    const Samples &samples, const Triangles &triangles,
    tf::fill::hole_chart_solve_scratch<Index> &scratch,
    tf::buffer<double> &residual) -> bool {
  const std::size_t n = collar.n_rim();
  const std::size_t count = samples.size();
  const std::size_t ring = std::size_t(collar.vertex_offsets[2]) - n;
  const std::size_t total = n + count + ring;
  const double scale = 1.0 / chart.radius;
  residual.clear();
  if (!count)
    return true;

  auto &positions = scratch.positions;
  auto &values = scratch.values;
  positions.allocate(total);
  values.allocate(total);
  const auto held = [&](std::size_t position, std::size_t slot) {
    positions[slot] = tf::point<double, 3>{chart.uv[position][0] * scale,
                                           chart.uv[position][1] * scale, 0.0};
    values[slot] = fixed[position];
  };
  for (std::size_t k = 0; k < n; ++k)
    held(k, k);
  for (std::size_t j = 0; j < count; ++j) {
    positions[n + j] =
        tf::point<double, 3>{samples[j][0] * scale, samples[j][1] * scale, 0.0};
    values[n + j] = 0.0;
  }
  for (std::size_t i = 0; i < ring; ++i)
    held(n + i, n + count + i);

  scratch.triangles.clear();
  for (const auto &triangle : triangles)
    scratch.triangles.push_back(
        {Index(triangle[0]), Index(triangle[1]), Index(triangle[2])});
  const auto lifted = [n, count](Index position) {
    return std::size_t(position) < n ? position : Index(position + count);
  };
  for (const auto &triangle : collar.ring_triangles)
    scratch.triangles.push_back(
        {lifted(triangle[0]), lifted(triangle[1]), lifted(triangle[2])});

  const auto polygons = tf::make_polygons(scratch.triangles, positions);
  scratch.membership.build(polygons);
  if (!tf::geometry::assemble_laplacian_system(
          polygons | tf::tag(scratch.membership),
          tf::geometry::cotangent_weights{}, scratch.assembly, scratch.system))
    return false;

  scratch.free_ids.allocate(count);
  scratch.free_of.allocate_and_initialize(total, Index(-1));
  for (std::size_t j = 0; j < count; ++j) {
    scratch.free_ids[j] = Index(n + j);
    scratch.free_of[n + j] = Index(j);
  }
  if (!tf::geometry::laplacian_is_anchored(scratch.system, scratch.free_ids,
                                           scratch.free_of, scratch.stack,
                                           scratch.seen))
    return false;
  tf::geometry::make_laplacian_form_pattern(
      scratch.system, scratch.free_ids, scratch.free_of, true, scratch.form);
  tf::geometry::state_laplacian_form_values(scratch.system, scratch.free_ids,
                                            scratch.free_of, 0.0, 1.0, 0.0,
                                            scratch.form);
  tf::geometry::state_dirichlet_load(scratch.form, values, 1, scratch.load);
  if (tf::geometry::solve_dirichlet_form(scratch.form, scratch.load, 1,
                                         scratch.solve, residual) !=
      tf::dirichlet_field_status::solved)
    return false;
  for (auto &value : residual)
    value *= chart.radius;
  return true;
}

} // namespace tf::fill

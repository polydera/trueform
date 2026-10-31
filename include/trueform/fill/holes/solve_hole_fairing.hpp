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
#include "../../core/algorithm/nth_element.hpp"
#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../geometry/dirichlet_field_status.hpp"
#include "../../geometry/laplacian/assemble_laplacian_system.hpp"
#include "../../geometry/laplacian/laplacian_is_anchored.hpp"
#include "../../geometry/laplacian/make_laplacian_form_pattern.hpp"
#include "../../geometry/laplacian/solve_dirichlet_form.hpp"
#include "../../geometry/laplacian/state_dirichlet_load.hpp"
#include "../../geometry/laplacian/state_laplacian_form_values.hpp"
#include "../../topology/policy/face_membership.hpp"
#include "./hole_fairing_frame.hpp"
#include "./hole_fairing_scratch.hpp"
#include "./hole_fairing_stencil.hpp"
#include <cmath>
#include <cstddef>

namespace tf::fill {

/// Fair one stencil, and state where its free vertices stand afterwards.
///
/// The solve runs in the stencil's own frame — its box's centre and diagonal
/// — so the form's two terms are weighed on one scale. The tether is what
/// makes the form definite where a run of free vertices reaches no fixed one,
/// pulling each free vertex toward where it stands; where every run does, none
/// is needed and none is charged. The form is factored once and answers all
/// three coordinates together, so a refusal leaves the refined patch whole.
///
/// `out` receives the free vertices' positions, in the stencil's own space
/// and in `free_ids` order.
template <typename Index>
auto solve_hole_fairing(const tf::fill::hole_fairing_stencil<Index> &stencil,
                        tf::fill::hole_fairing_scratch<Index> &scratch,
                        tf::buffer<tf::point<double, 3>> &out) -> bool {
  const std::size_t n = stencil.positions.size();
  const std::size_t free_count = stencil.free_ids.size();
  out.clear();
  if (!free_count)
    return true;

  tf::point<double, 3> centre{0.0, 0.0, 0.0};
  double scale = 0.0;
  if (!tf::fill::hole_fairing_frame(stencil.positions, centre, scale))
    return false;
  scratch.local.allocate(n);
  for (std::size_t k = 0; k < n; ++k)
    for (std::size_t axis = 0; axis < 3; ++axis)
      scratch.local[k][axis] =
          (stencil.positions[k][axis] - centre[axis]) / scale;

  const auto polygons = tf::make_polygons(stencil.triangles, scratch.local);
  scratch.membership.build(polygons);
  if (!tf::geometry::assemble_laplacian_system(
          polygons | tf::tag(scratch.membership),
          tf::geometry::cotangent_weights{}, scratch.assembly, scratch.system))
    return false;

  auto &form = scratch.form;
  tf::geometry::make_laplacian_form_pattern(scratch.system, stencil.free_ids,
                                            stencil.free_of, true, form);
  tf::geometry::state_laplacian_form_values(
      scratch.system, stencil.free_ids, stencil.free_of,
      tf::fill::hole_fairing_stiffness, tf::fill::hole_fairing_bending, 0.0,
      form);
  scratch.held.allocate(3 * n);
  for (std::size_t k = 0; k < n; ++k)
    for (std::size_t axis = 0; axis < 3; ++axis)
      scratch.held[3 * k + axis] = scratch.local[k][axis];
  tf::geometry::state_dirichlet_load(form, scratch.held, 3, scratch.load);

  // A row holds its diagonal last.
  const auto diagonal = [&form](std::size_t k) -> double & {
    return form.free.values[std::size_t(form.free.row_offsets[k + 1]) - 1];
  };
  if (!tf::geometry::laplacian_is_anchored(scratch.system, stencil.free_ids,
                                           stencil.free_of, scratch.stack,
                                           scratch.seen)) {
    scratch.medians.allocate(free_count);
    for (std::size_t k = 0; k < free_count; ++k)
      scratch.medians[k] =
          diagonal(k) / scratch.system.mass[std::size_t(stencil.free_ids[k])];
    const auto median = scratch.medians.begin() + (free_count - 1) / 2;
    tf::nth_element(scratch.medians.begin(), median, scratch.medians.end());
    const double tether = 1e-4 * *median;
    if (!(tether > 0.0) || !std::isfinite(tether))
      return false;
    for (std::size_t k = 0; k < free_count; ++k) {
      const std::size_t row = std::size_t(stencil.free_ids[k]);
      const double pull = tether * scratch.system.mass[row];
      diagonal(k) += pull;
      for (std::size_t axis = 0; axis < 3; ++axis)
        scratch.load[3 * k + axis] += pull * scratch.local[row][axis];
    }
  }

  if (tf::geometry::solve_dirichlet_form(form, scratch.load, 3, scratch.solve,
                                         scratch.solved) !=
      tf::dirichlet_field_status::solved)
    return false;
  out.allocate(free_count);
  for (std::size_t k = 0; k < free_count; ++k)
    for (std::size_t axis = 0; axis < 3; ++axis) {
      out[k][axis] = scratch.solved[3 * k + axis] * scale + centre[axis];
      if (!std::isfinite(out[k][axis])) {
        out.clear();
        return false;
      }
    }
  return true;
}

} // namespace tf::fill

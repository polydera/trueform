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
#include "./apply_hole_fairing_operator.hpp"
#include "./hole_fairing_is_anchored.hpp"
#include "./hole_fairing_stencil.hpp"
#include "./hole_fairing_system.hpp"
#include "./solve_hole_fairing_axis.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tf::fill {

/// Fair one stencil, and state where its free vertices stand afterwards.
///
/// The solve runs in the stencil's own frame — its box's centre and diagonal
/// — so the form's two terms are weighed on one scale. The tether is what
/// makes the form definite where a run of free vertices reaches no fixed one;
/// where every run does, none is needed and none is charged. All three
/// coordinates are answered before any of them is stated, so a refusal in one
/// leaves the refined patch whole.
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

  if (!tf::fill::assemble_hole_fairing_system(
          stencil.triangles, scratch.local, scratch.assembly, scratch.system))
    return false;

  double tether = 0.0;
  if (!tf::fill::hole_fairing_is_anchored(scratch.system, stencil.free_ids,
                                          stencil.free_of, scratch.stack,
                                          scratch.seen)) {
    scratch.medians.clear();
    for (auto id : stencil.free_ids) {
      const std::size_t row = std::size_t(id);
      scratch.medians.push_back(
          tf::fill::hole_fairing_form_diagonal(
              scratch.system, tf::fill::hole_fairing_stiffness,
              tf::fill::hole_fairing_bending, row) /
          scratch.system.mass[row]);
    }
    std::sort(scratch.medians.begin(), scratch.medians.end());
    tether = 1e-4 * scratch.medians[(free_count - 1) / 2];
    if (!std::isfinite(tether))
      return false;
  }

  scratch.jacobi.allocate(free_count);
  for (std::size_t k = 0; k < free_count; ++k) {
    const std::size_t row = std::size_t(stencil.free_ids[k]);
    scratch.jacobi[k] = tf::fill::hole_fairing_form_diagonal(
                            scratch.system, tf::fill::hole_fairing_stiffness,
                            tf::fill::hole_fairing_bending, row) +
                        tether * scratch.system.mass[row];
    if (!std::isfinite(scratch.jacobi[k]) || scratch.jacobi[k] <= 0.0)
      return false;
  }

  out.allocate(free_count);
  for (std::size_t k = 0; k < free_count; ++k)
    out[k] = scratch.local[std::size_t(stencil.free_ids[k])];
  for (std::size_t axis = 0; axis < 3; ++axis) {
    scratch.coordinate.allocate(n);
    for (std::size_t k = 0; k < n; ++k)
      scratch.coordinate[k] = scratch.local[k][axis];
    tf::fill::hole_fairing_form(
        scratch.system, tf::fill::hole_fairing_stiffness,
        tf::fill::hole_fairing_bending, scratch.coordinate, scratch.form,
        scratch.form.stated);
    scratch.rhs.allocate(free_count);
    for (std::size_t k = 0; k < free_count; ++k)
      scratch.rhs[k] = -scratch.form.stated[std::size_t(stencil.free_ids[k])];
    if (!tf::fill::solve_hole_fairing_axis(scratch.system, stencil.free_ids,
                                           tether, scratch,
                                           scratch.displacement)) {
      out.clear();
      return false;
    }
    for (std::size_t k = 0; k < free_count; ++k)
      out[k][axis] += scratch.displacement[k];
  }

  for (std::size_t k = 0; k < free_count; ++k)
    for (std::size_t axis = 0; axis < 3; ++axis) {
      out[k][axis] = out[k][axis] * scale + centre[axis];
      if (!std::isfinite(out[k][axis])) {
        out.clear();
        return false;
      }
    }
  return true;
}

} // namespace tf::fill

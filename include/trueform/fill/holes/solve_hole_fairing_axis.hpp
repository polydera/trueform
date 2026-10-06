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
#include "./hole_fairing_stencil.hpp"
#include "./hole_fairing_system.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tf::fill {

/// What the fairing's stiffness weighs against its bending.
inline constexpr double hole_fairing_stiffness = 0.1;
/// What the fairing's bending weighs against its stiffness.
inline constexpr double hole_fairing_bending = 1.0;

/// The scratch one group's fairing walks on, the operator it assembles
/// included, so a worker carries one solve's memory across the groups it
/// takes.
template <typename Index> struct hole_fairing_scratch {
  tf::fill::hole_fairing_stencil_scratch<Index> gather;
  tf::fill::hole_fairing_assembly_scratch<Index> assembly;
  tf::fill::hole_fairing_system<Index> system;
  tf::fill::hole_fairing_form_scratch form;
  tf::buffer<tf::point<double, 3>> local;
  tf::buffer<double> coordinate;
  tf::buffer<double> rhs;
  tf::buffer<double> jacobi;
  tf::buffer<double> displacement;
  tf::buffer<double> residual;
  tf::buffer<double> direction;
  tf::buffer<double> preconditioned;
  tf::buffer<double> applied;
  tf::buffer<double> medians;
  tf::buffer<Index> stack;
  tf::buffer<char> seen;
};

/// Solve one axis of the displacement form, and state whether the answer
/// holds.
///
/// The preconditioned conjugate gradient starts at no displacement, restates
/// its residual every fiftieth step, and is accepted only against the true
/// residual recomputed with the same operator it ran on — so a direction that
/// drifted is caught rather than believed. A curvature that is not positive,
/// or any quantity that stops evaluating finitely, refuses outright.
template <typename Index>
auto solve_hole_fairing_axis(const tf::fill::hole_fairing_system<Index> &system,
                             const tf::buffer<Index> &free_ids, double tether,
                             tf::fill::hole_fairing_scratch<Index> &scratch,
                             tf::buffer<double> &out) -> bool {
  const std::size_t n = free_ids.size();
  const auto dot = [n](const tf::buffer<double> &a,
                       const tf::buffer<double> &b) {
    double value = 0.0;
    for (std::size_t k = 0; k < n; ++k)
      value += a[k] * b[k];
    return value;
  };
  const auto apply = [&system, &free_ids, tether,
                      &scratch](const tf::buffer<double> &in,
                                tf::buffer<double> &result) {
    tf::fill::apply_hole_fairing_operator(
        system, tf::fill::hole_fairing_stiffness,
        tf::fill::hole_fairing_bending, tether, free_ids, in, scratch.form,
        result);
  };

  out.allocate(n);
  for (std::size_t k = 0; k < n; ++k)
    out[k] = 0.0;
  const double norm_b = std::sqrt(dot(scratch.rhs, scratch.rhs));
  if (!std::isfinite(norm_b))
    return false;
  const double tolerance = std::max(1e-12, 1e-8 * norm_b);

  scratch.residual.allocate(n);
  scratch.preconditioned.allocate(n);
  scratch.direction.allocate(n);
  for (std::size_t k = 0; k < n; ++k) {
    scratch.residual[k] = scratch.rhs[k];
    scratch.preconditioned[k] = scratch.residual[k] / scratch.jacobi[k];
    scratch.direction[k] = scratch.preconditioned[k];
  }
  double rz = dot(scratch.residual, scratch.preconditioned);
  const std::size_t cap = n > 500 ? 256 : 128;
  for (std::size_t iteration = 0; iteration < cap; ++iteration) {
    if (std::sqrt(dot(scratch.residual, scratch.residual)) <= tolerance)
      break;
    apply(scratch.direction, scratch.applied);
    const double curvature = dot(scratch.direction, scratch.applied);
    if (!std::isfinite(curvature) || curvature <= 0.0)
      return false;
    const double step = rz / curvature;
    if (!std::isfinite(step))
      return false;
    for (std::size_t k = 0; k < n; ++k)
      out[k] += step * scratch.direction[k];
    if ((iteration + 1) % 50 == 0) {
      apply(out, scratch.applied);
      for (std::size_t k = 0; k < n; ++k)
        scratch.residual[k] = scratch.rhs[k] - scratch.applied[k];
    } else {
      for (std::size_t k = 0; k < n; ++k)
        scratch.residual[k] -= step * scratch.applied[k];
    }
    for (std::size_t k = 0; k < n; ++k)
      scratch.preconditioned[k] = scratch.residual[k] / scratch.jacobi[k];
    const double next = dot(scratch.residual, scratch.preconditioned);
    if (!std::isfinite(next) || rz == 0.0)
      return false;
    const double blend = next / rz;
    for (std::size_t k = 0; k < n; ++k)
      scratch.direction[k] =
          scratch.preconditioned[k] + blend * scratch.direction[k];
    rz = next;
  }

  apply(out, scratch.applied);
  double stated = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    const double value = scratch.rhs[k] - scratch.applied[k];
    stated += value * value;
  }
  if (!std::isfinite(stated) || std::sqrt(stated) > tolerance)
    return false;
  for (std::size_t k = 0; k < n; ++k)
    if (!std::isfinite(out[k]))
      return false;
  return true;
}

} // namespace tf::fill

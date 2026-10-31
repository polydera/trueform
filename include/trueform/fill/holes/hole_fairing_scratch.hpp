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
#include "../../geometry/laplacian/assemble_laplacian_system.hpp"
#include "../../geometry/laplacian/laplacian_form.hpp"
#include "../../geometry/laplacian/laplacian_system.hpp"
#include "../../geometry/laplacian/solve_dirichlet_form.hpp"
#include "../../topology/face_membership.hpp"
#include "./hole_fairing_stencil.hpp"

namespace tf::fill {

/// What the fairing's stiffness weighs against its bending.
inline constexpr double hole_fairing_stiffness = 0.1;
/// What the fairing's bending weighs against its stiffness.
inline constexpr double hole_fairing_bending = 1.0;

/// The scratch one group's fairing walks on, the operator it assembles and
/// factors included, so a worker carries one solve's memory across the
/// groups it takes.
template <typename Index> struct hole_fairing_scratch {
  tf::fill::hole_fairing_stencil_scratch<Index> gather;
  tf::face_membership<Index> membership;
  tf::geometry::laplacian_assembly_scratch assembly;
  tf::geometry::laplacian_system<Index> system;
  tf::geometry::laplacian_form<Index> form;
  tf::geometry::dirichlet_solve_scratch<Index> solve;
  tf::buffer<tf::point<double, 3>> local;
  tf::buffer<double> held;
  tf::buffer<double> load;
  tf::buffer<double> solved;
  tf::buffer<double> medians;
  tf::buffer<Index> stack;
  tf::buffer<char> seen;
};

} // namespace tf::fill

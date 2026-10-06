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
#include "./hole_fairing_system.hpp"
#include <cstddef>

namespace tf::fill {

/// `out = L * in` over the whole stencil, every row summed over its own
/// ascending columns.
template <typename Index>
auto hole_fairing_laplacian(const tf::fill::hole_fairing_system<Index> &system,
                            const tf::buffer<double> &in,
                            tf::buffer<double> &out) -> void {
  const std::size_t n = system.size();
  out.allocate(n);
  for (std::size_t k = 0; k < n; ++k) {
    double value = system.diagonal[k] * in[k];
    for (Index slot = system.row_offsets[k]; slot < system.row_offsets[k + 1];
         ++slot)
      value += system.weights[std::size_t(slot)] *
               in[std::size_t(system.columns[std::size_t(slot)])];
    out[k] = value;
  }
}

/// The scratch the form's passes stand on.
struct hole_fairing_form_scratch {
  tf::buffer<double> full;
  tf::buffer<double> stretch;
  tf::buffer<double> lumped;
  tf::buffer<double> bend;
  tf::buffer<double> stated;
};

/// `out = (k_s * L + k_b * L * M^-1 * L) * in` over the whole stencil.
template <typename Index>
auto hole_fairing_form(const tf::fill::hole_fairing_system<Index> &system,
                       double stiffness, double bending,
                       const tf::buffer<double> &in,
                       tf::fill::hole_fairing_form_scratch &scratch,
                       tf::buffer<double> &out) -> void {
  const std::size_t n = system.size();
  tf::fill::hole_fairing_laplacian(system, in, scratch.stretch);
  scratch.lumped.allocate(n);
  for (std::size_t k = 0; k < n; ++k)
    scratch.lumped[k] = scratch.stretch[k] / system.mass[k];
  tf::fill::hole_fairing_laplacian(system, scratch.lumped, scratch.bend);
  out.allocate(n);
  for (std::size_t k = 0; k < n; ++k)
    out[k] = stiffness * scratch.stretch[k] + bending * scratch.bend[k];
}

/// `out = (H_FF + tau * M_F) * in` over the free rows, the displacement
/// extended by zeros onto the fixed ones.
template <typename Index>
auto apply_hole_fairing_operator(
    const tf::fill::hole_fairing_system<Index> &system, double stiffness,
    double bending, double tether, const tf::buffer<Index> &free_ids,
    const tf::buffer<double> &in,
    tf::fill::hole_fairing_form_scratch &scratch, tf::buffer<double> &out)
    -> void {
  const std::size_t n = system.size();
  scratch.full.allocate(n);
  for (std::size_t k = 0; k < n; ++k)
    scratch.full[k] = 0.0;
  for (std::size_t k = 0; k < free_ids.size(); ++k)
    scratch.full[std::size_t(free_ids[k])] = in[k];
  tf::fill::hole_fairing_form(system, stiffness, bending, scratch.full, scratch,
                              scratch.stated);
  out.allocate(free_ids.size());
  for (std::size_t k = 0; k < free_ids.size(); ++k) {
    const std::size_t row = std::size_t(free_ids[k]);
    out[k] = scratch.stated[row] + tether * system.mass[row] * in[k];
  }
}

/// `H_ii`, the form's own diagonal, which every fixed neighbour states into.
template <typename Index>
auto hole_fairing_form_diagonal(
    const tf::fill::hole_fairing_system<Index> &system, double stiffness,
    double bending, std::size_t row) -> double {
  double bend = system.diagonal[row] * system.diagonal[row] /
                system.mass[row];
  for (Index slot = system.row_offsets[row];
       slot < system.row_offsets[row + 1]; ++slot)
    bend += system.weights[std::size_t(slot)] *
            system.weights[std::size_t(slot)] /
            system.mass[std::size_t(system.columns[std::size_t(slot)])];
  return stiffness * system.diagonal[row] + bending * bend;
}

} // namespace tf::fill

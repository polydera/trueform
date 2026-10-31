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
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/buffer.hpp"
#include "../../linalg/analyze_cholesky.hpp"
#include "../../linalg/cholesky/refine_once.hpp"
#include "../../linalg/cholesky_factor.hpp"
#include "../../linalg/cholesky_pattern.hpp"
#include "../../linalg/cholesky_scratch.hpp"
#include "../../linalg/factor_cholesky.hpp"
#include "../../linalg/solve_cholesky.hpp"
#include "../dirichlet_field_status.hpp"
#include "./laplacian_form.hpp"
#include <cstddef>

namespace tf::geometry {

/// The scratch one Dirichlet solve walks on, the factor it builds included.
template <typename Index> struct dirichlet_solve_scratch {
  tf::cholesky_pattern<Index> pattern;
  tf::cholesky_factor factor;
  tf::cholesky_scratch<Index> cholesky;
};

/// Solve the free rows of `r` fields the form holds at rest, the load each
/// free row carries stated by `rhs`, `r` lanes per free row.
///
/// The free block is factored once by sparse Cholesky and serves every lane,
/// and the answer is refined once and accepted against its true residual
/// (@ref tf::linalg::refine_once). The caller has proven the free block
/// definite — a positive tether, or @ref tf::geometry::laplacian_is_anchored —
/// and a pivot that still is not positive refuses.
///
/// `out` receives `r` values per free row, in free order. Answers `solved`,
/// or the solve's own refusal: `refused_index`, `refused_indefinite` or
/// `refused_residual`.
template <typename Index>
auto solve_dirichlet_form(const tf::geometry::laplacian_form<Index> &form,
                          const tf::buffer<double> &rhs, std::size_t r,
                          tf::geometry::dirichlet_solve_scratch<Index> &scratch,
                          tf::buffer<double> &out)
    -> tf::dirichlet_field_status {
  if (!tf::analyze_cholesky(form.free, scratch.cholesky, scratch.pattern))
    return tf::dirichlet_field_status::refused_index;
  if (!tf::factor_cholesky(scratch.pattern, form.free.values, scratch.cholesky,
                           scratch.factor))
    return tf::dirichlet_field_status::refused_indefinite;
  out.allocate(rhs.size());
  tf::parallel_copy(rhs, out);
  tf::solve_cholesky(scratch.pattern, scratch.factor, out, r, scratch.cholesky);
  if (!tf::linalg::refine_once(form.free, scratch.pattern, scratch.factor, rhs,
                               r, scratch.cholesky, out))
    return tf::dirichlet_field_status::refused_residual;
  return tf::dirichlet_field_status::solved;
}

} // namespace tf::geometry

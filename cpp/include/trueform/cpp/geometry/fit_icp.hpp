/*
 * Copyright (c) 2025 XLAB
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

#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/point_cloud.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Options for iterative closest-point alignment.
template <typename Real> struct fit_icp_options {
  int max_iterations = 100;
  int n_samples = 1000;
  int k = 1;
  Real sigma = Real{-1};
  Real outlier_proportion = Real{0};
  Real min_relative_improvement = static_cast<Real>(1e-6F);
  Real ema_alpha = static_cast<Real>(0.3F);
};

/// @brief Fit an iterative closest-point delta transformation from source to
/// target. Each cloud is read in its own frame, with the normals it carries.
/// The returned array has shape [Dims + 1, Dims + 1].
template <typename Real, std::size_t Dims = 3>
auto fit_icp(const point_cloud<Real, Dims> &source,
             const point_cloud<Real, Dims> &target,
             const fit_icp_options<Real> &options = {}) -> nd_array<Real>;

#define TF_CPP_EXTERN_FIT_ICP(Real, Dims)                                      \
  extern template auto fit_icp<Real, Dims>(                                    \
      const point_cloud<Real, Dims> &, const point_cloud<Real, Dims> &,        \
      const fit_icp_options<Real> &) -> nd_array<Real>

TF_CPP_MATRIX_FOR_EACH_REAL_DIMS(TF_CPP_EXTERN_FIT_ICP)

#undef TF_CPP_EXTERN_FIT_ICP

} // namespace tf::cpp

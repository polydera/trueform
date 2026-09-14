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

/// @brief Options for one k-nearest-neighbor alignment step.
/// Neighbor counts above ten use the core kernel's bounded ten-neighbor
/// scratch. A negative sigma selects adaptive width; zero falls back to the
/// nearest correspondence when the effective Gaussian width is zero.
template <typename Real> struct fit_knn_options {
  int k = 1;
  Real sigma = Real{-1};
  Real outlier_proportion = Real{0};
};

/// @brief Fit one k-nearest-neighbor alignment delta from source to target.
/// In 3D, target normals select point-to-plane fitting and optional source
/// normals weight that fit. In 2D, normals are not supported. The returned
/// array has shape [Dims + 1, Dims + 1].
template <typename Real, std::size_t Dims = 3>
auto fit_knn(const point_cloud<Real, Dims> &source,
             const point_cloud<Real, Dims> &target,
             const fit_knn_options<Real> &options = {}) -> nd_array<Real>;

#define TF_CPP_EXTERN_FIT_KNN(Real, Dims)                                      \
  extern template auto fit_knn<Real, Dims>(                                    \
      const point_cloud<Real, Dims> &, const point_cloud<Real, Dims> &,        \
      const fit_knn_options<Real> &) -> nd_array<Real>

TF_CPP_MATRIX_FOR_EACH_REAL_DIMS(TF_CPP_EXTERN_FIT_KNN)

#undef TF_CPP_EXTERN_FIT_KNN

} // namespace tf::cpp

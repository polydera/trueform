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
#include "trueform/cpp/core/point_cloud.hpp"
#include "trueform/cpp/geometry/chamfer_error_options.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Compute the one-way mean nearest-neighbor distance from source to
/// target.
template <typename Real, std::size_t Dims = 3>
auto chamfer_error(const point_cloud<Real, Dims> &source,
                   const point_cloud<Real, Dims> &target,
                   const chamfer_error_options<Real> &options = {}) -> Real;

#define TF_CPP_EXTERN_CHAMFER_ERROR(Real, Dims)                                \
  extern template auto chamfer_error<Real, Dims>(                              \
      const point_cloud<Real, Dims> &, const point_cloud<Real, Dims> &,        \
      const chamfer_error_options<Real> &) -> Real

TF_CPP_MATRIX_FOR_EACH_REAL_DIMS(TF_CPP_EXTERN_CHAMFER_ERROR)

#undef TF_CPP_EXTERN_CHAMFER_ERROR

} // namespace tf::cpp

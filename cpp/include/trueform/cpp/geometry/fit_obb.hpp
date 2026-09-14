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

/// @brief Options for oriented-bounding-box alignment.
struct fit_obb_options {
  int sample_size = 100;
};

/// @brief Fit a coarse oriented-bounding-box delta transformation from source
/// to target. Each cloud is read in its own frame. The returned array has
/// shape [Dims + 1, Dims + 1].
template <typename Real, std::size_t Dims = 3>
auto fit_obb(const point_cloud<Real, Dims> &source,
             const point_cloud<Real, Dims> &target,
             const fit_obb_options &options = {}) -> nd_array<Real>;

#define TF_CPP_EXTERN_FIT_OBB(Real, Dims)                                      \
  extern template auto fit_obb<Real, Dims>(                                    \
      const point_cloud<Real, Dims> &, const point_cloud<Real, Dims> &,        \
      const fit_obb_options &) -> nd_array<Real>

TF_CPP_MATRIX_FOR_EACH_REAL_DIMS(TF_CPP_EXTERN_FIT_OBB)

#undef TF_CPP_EXTERN_FIT_OBB

} // namespace tf::cpp

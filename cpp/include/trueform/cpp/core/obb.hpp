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

namespace tf::cpp {

/// @brief Typed input for computing a three-dimensional oriented bounding box.
template <typename Real> struct obb_options {
  nd_array<Real> points;
};

/// @brief Native oriented bounding box arrays.
template <typename Real> struct obb_result {
  nd_array<Real> origin;
  nd_array<Real> axes;
  nd_array<Real> extent;
};

/// @brief Compute a PCA-based oriented bounding box from three-dimensional
/// points.
template <typename Real>
auto obb_from(const obb_options<Real> &options) -> obb_result<Real>;

#define TF_CPP_EXTERN_OBB(Real)                                                \
  extern template auto obb_from(const obb_options<Real> &) -> obb_result<Real>

TF_CPP_MATRIX_FOR_EACH_REAL(TF_CPP_EXTERN_OBB)

#undef TF_CPP_EXTERN_OBB

} // namespace tf::cpp

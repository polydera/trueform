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
#include "trueform/cpp/spatial/primitive.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Apply an affine transformation to a runtime primitive.
///
/// The matrix must have shape [Dims + 1, Dims + 1]. Points and primitive
/// vertices include translation, while vectors do not. Ray and line directions
/// are normalized after their linear transformation. Plane normals use the
/// inverse transpose. The returned primitive owns detached coordinate storage
/// and preserves the input kind, cardinality, count, and raw array shape.
template <typename Real, std::size_t Dims>
auto transformed(const primitive<Real, Dims> &value,
                 const nd_array<Real> &matrix) -> primitive<Real, Dims>;

#define TF_CPP_EXTERN_TRANSFORMED(Real, Dims)                                  \
  extern template auto transformed<Real, Dims>(const primitive<Real, Dims> &,  \
                                               const nd_array<Real> &)         \
      -> primitive<Real, Dims>

TF_CPP_MATRIX_FOR_EACH_REAL_DIMS(TF_CPP_EXTERN_TRANSFORMED)

#undef TF_CPP_EXTERN_TRANSFORMED

} // namespace tf::cpp

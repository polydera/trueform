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

#include "trueform/core/transformation_like.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>
#include <utility>

namespace tf::cpp {

/// @brief Convert an affine transformation to a full square matrix.
///
/// The returned array has shape [(Dims + 1), (Dims + 1)] and element type
/// Real. The implicit final affine row is materialized as [0, ..., 0, 1].
template <typename Real, std::size_t Dims, typename Policy>
auto to_matrix(const tf::transformation_like<Dims, Policy> &transformation)
    -> nd_array<Real> {
  constexpr auto size = Dims + 1;
  tf::buffer<Real> buffer;
  buffer.allocate(size * size);
  for (std::size_t row = 0; row < Dims; ++row)
    for (std::size_t column = 0; column < size; ++column)
      buffer[row * size + column] =
          static_cast<Real>(transformation(row, column));
  for (std::size_t column = 0; column < Dims; ++column)
    buffer[Dims * size + column] = Real{0};
  buffer[Dims * size + Dims] = Real{1};
  return nd_array<Real>::from_buffer(
      std::move(buffer), {static_cast<int>(size), static_cast<int>(size)});
}

#define TF_CPP_EXTERN_TO_MATRIX(Real, Dims)                                    \
  extern template auto to_matrix<Real, Dims, tf::linalg::trans<Real, Dims>>(   \
      const tf::transformation_like<Dims, tf::linalg::trans<Real, Dims>> &)    \
      -> nd_array<Real>

TF_CPP_MATRIX_FOR_EACH_REAL_DIMS(TF_CPP_EXTERN_TO_MATRIX)

#undef TF_CPP_EXTERN_TO_MATRIX

} // namespace tf::cpp

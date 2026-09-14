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
#include "trueform/cpp/core/to_matrix.hpp"

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_TO_MATRIX(Real, Dims)                               \
  template auto to_matrix<Real, Dims, tf::linalg::trans<Real, Dims>>(          \
      const tf::transformation_like<Dims, tf::linalg::trans<Real, Dims>> &)    \
      -> nd_array<Real>

TF_CPP_MATRIX_FOR_EACH_REAL_DIMS(TF_CPP_INSTANTIATE_TO_MATRIX)

#undef TF_CPP_INSTANTIATE_TO_MATRIX

} // namespace tf::cpp

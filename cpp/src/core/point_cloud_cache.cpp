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
#include "./point_cloud_cache_impl.hpp"

#include "trueform/cpp/core/matrix.hpp"

namespace tf::cpp {
#define TF_CPP_INSTANTIATE_POINT_CLOUD_CACHE(Real, Dims)                       \
  template class point_cloud_cache<Real, Dims>

TF_CPP_MATRIX_FOR_EACH_REAL_DIMS(TF_CPP_INSTANTIATE_POINT_CLOUD_CACHE)

#undef TF_CPP_INSTANTIATE_POINT_CLOUD_CACHE
} // namespace tf::cpp

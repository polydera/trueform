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

#ifndef TF_CPP_RAY_CAST_FORM_REAL
#error "TF_CPP_RAY_CAST_FORM_REAL is required"
#endif
#ifndef TF_CPP_RAY_CAST_DIMS
#error "TF_CPP_RAY_CAST_DIMS is required"
#endif

#include "ray_cast_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_POINT_CLOUD_RAY_CAST(RayReal)                       \
  template auto                                                                \
  ray_cast<RayReal, TF_CPP_RAY_CAST_FORM_REAL, TF_CPP_RAY_CAST_DIMS>(          \
      const primitive<RayReal, TF_CPP_RAY_CAST_DIMS> &,                        \
      const point_cloud<TF_CPP_RAY_CAST_FORM_REAL, TF_CPP_RAY_CAST_DIMS> &,    \
      const ray_cast_options<TF_CPP_RAY_CAST_FORM_REAL> &)                     \
      -> ray_cast_form_result<std::int32_t, TF_CPP_RAY_CAST_FORM_REAL>

TF_CPP_MATRIX_FOR_EACH_REAL(TF_CPP_INSTANTIATE_POINT_CLOUD_RAY_CAST)

#undef TF_CPP_INSTANTIATE_POINT_CLOUD_RAY_CAST

} // namespace tf::cpp

#undef TF_CPP_RAY_CAST_DIMS
#undef TF_CPP_RAY_CAST_FORM_REAL

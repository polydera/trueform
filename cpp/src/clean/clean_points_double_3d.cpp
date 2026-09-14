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
#include "clean_impl.hpp"

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_CLEAN_POINTS(Real, Dims)                            \
  template auto cleaned_points<Real, Dims>(const nd_array<Real> &, Real, bool, \
                                           bool) -> nd_array<Real>;            \
  template auto cleaned_points_with_map<Real, Dims>(const nd_array<Real> &,    \
                                                    Real, bool, bool)          \
      -> cleaned_points_result<default_index_t, Real>;                         \
  template auto cleaned_points<Real, Dims>(                                    \
      const point_cloud<Real, Dims> &, Real, bool, bool) -> nd_array<Real>;    \
  template auto cleaned_points_with_map<Real, Dims>(                           \
      const point_cloud<Real, Dims> &, Real, bool, bool)                       \
      -> cleaned_points_result<default_index_t, Real>

TF_CPP_INSTANTIATE_CLEAN_POINTS(double, 3);

#undef TF_CPP_INSTANTIATE_CLEAN_POINTS

} // namespace tf::cpp

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
#include "isocontours_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_ISOCONTOURS(Ngon)                                   \
  template auto isocontours<std::int32_t, double, Ngon>(                       \
      const mesh<std::int32_t, double, 3, Ngon> &, const nd_array<double> &,   \
      double) -> tf::curves_buffer<std::int32_t, double, 3>;                   \
  template auto isocontours<std::int32_t, double, Ngon>(                       \
      const mesh<std::int32_t, double, 3, Ngon> &, const nd_array<double> &,   \
      const nd_array<double> &) -> tf::curves_buffer<std::int32_t, double, 3>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_ISOCONTOURS)

#undef TF_CPP_INSTANTIATE_ISOCONTOURS

} // namespace tf::cpp

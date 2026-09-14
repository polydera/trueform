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
#include "orientation_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_ORIENTATION(Ngon)                                   \
  template auto reverse_winding<std::int64_t, double, 2, Ngon>(                \
      const mesh<std::int64_t, double, 2, Ngon> &)                             \
      -> tf::polygons_buffer<std::int64_t, double, 2, Ngon>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_ORIENTATION)

#undef TF_CPP_INSTANTIATE_ORIENTATION

} // namespace tf::cpp

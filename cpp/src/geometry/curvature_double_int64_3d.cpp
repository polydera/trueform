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
#include "curvature_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_CURVATURE(Ngon)                                     \
  template auto principal_curvatures<std::int64_t, double, 3, Ngon>(           \
      const mesh<std::int64_t, double, 3, Ngon> &, int)                        \
      -> principal_curvatures_result<double>;                                  \
  template auto principal_directions<std::int64_t, double, 3, Ngon>(           \
      const mesh<std::int64_t, double, 3, Ngon> &, int)                        \
      -> principal_directions_result<double>;                                  \
  template auto shape_index<std::int64_t, double, 3, Ngon>(                    \
      const mesh<std::int64_t, double, 3, Ngon> &, int) -> nd_array<double>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_CURVATURE)

#undef TF_CPP_INSTANTIATE_CURVATURE

} // namespace tf::cpp

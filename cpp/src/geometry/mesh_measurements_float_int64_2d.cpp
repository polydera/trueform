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
#include "mesh_measurements_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_MESH_MEASUREMENTS(Ngon)                             \
  template auto area<std::int64_t, float, 2, Ngon>(                            \
      const mesh<std::int64_t, float, 2, Ngon> &) -> float;                    \
  template auto mean_edge_length<std::int64_t, float, 2, Ngon>(                \
      const mesh<std::int64_t, float, 2, Ngon> &) -> float;                    \
  template auto min_edge_length<std::int64_t, float, 2, Ngon>(                 \
      const mesh<std::int64_t, float, 2, Ngon> &) -> float;                    \
  template auto max_edge_length<std::int64_t, float, 2, Ngon>(                 \
      const mesh<std::int64_t, float, 2, Ngon> &) -> float

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_MESH_MEASUREMENTS)

#undef TF_CPP_INSTANTIATE_MESH_MEASUREMENTS
} // namespace tf::cpp

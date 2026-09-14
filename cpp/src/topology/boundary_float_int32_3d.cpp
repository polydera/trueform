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
#include "boundary_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_MESH_BOUNDARY(Ngon)                                 \
  template auto boundary_edges<std::int32_t, float, 3, Ngon>(                  \
      const mesh<std::int32_t, float, 3, Ngon> &) -> nd_array<std::int32_t>;   \
  template auto boundary_paths<std::int32_t, float, 3, Ngon>(                  \
      const mesh<std::int32_t, float, 3, Ngon> &)                              \
      -> offset_blocked_buffer<std::int32_t, std::int32_t>;                    \
  template auto boundary_curves<std::int32_t, float, 3, Ngon>(                 \
      const mesh<std::int32_t, float, 3, Ngon> &)                              \
      -> boundary_curves_result<std::int32_t, float, 3>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_MESH_BOUNDARY)

#undef TF_CPP_INSTANTIATE_MESH_BOUNDARY

} // namespace tf::cpp

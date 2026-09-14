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
#include "normals_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_MESH_NORMALS(Ngon)                                  \
  template auto normals<std::int64_t, float, 3, Ngon>(                         \
      const mesh<std::int64_t, float, 3, Ngon> &) -> nd_array<float>;          \
  template auto point_normals<std::int64_t, float, 3, Ngon>(                   \
      const mesh<std::int64_t, float, 3, Ngon> &) -> nd_array<float>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_MESH_NORMALS)

#undef TF_CPP_INSTANTIATE_MESH_NORMALS

} // namespace tf::cpp

/*
 * Copyright (c) 2026 XLAB
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
#include "quality_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_MESH_QUALITY(Ngon)                                  \
  template auto face_quality<std::int32_t, double, 3, Ngon>(                   \
      const mesh<std::int32_t, double, 3, Ngon> &)                             \
      -> face_quality_result<double>;                                          \
  template auto dihedral_angles<std::int32_t, double, 3, Ngon>(                \
      const mesh<std::int32_t, double, 3, Ngon> &)                             \
      -> dihedral_angles_result<std::int32_t, double>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_MESH_QUALITY)

#undef TF_CPP_INSTANTIATE_MESH_QUALITY

} // namespace tf::cpp

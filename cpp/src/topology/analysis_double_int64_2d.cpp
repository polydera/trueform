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
#include "topology_analysis_impl.hpp"

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_MESH_ANALYSIS(Ngon)                                 \
  template auto is_closed<std::int64_t, double, 2, Ngon>(                      \
      const mesh<std::int64_t, double, 2, Ngon> &) -> bool;                    \
  template auto is_open<std::int64_t, double, 2, Ngon>(                        \
      const mesh<std::int64_t, double, 2, Ngon> &) -> bool;                    \
  template auto is_manifold<std::int64_t, double, 2, Ngon>(                    \
      const mesh<std::int64_t, double, 2, Ngon> &) -> bool;                    \
  template auto is_non_manifold<std::int64_t, double, 2, Ngon>(                \
      const mesh<std::int64_t, double, 2, Ngon> &) -> bool;                    \
  template auto euler_characteristic<std::int64_t, double, 2, Ngon>(           \
      const mesh<std::int64_t, double, 2, Ngon> &) -> std::int32_t;            \
  template auto non_manifold_edges<std::int64_t, double, 2, Ngon>(             \
      const mesh<std::int64_t, double, 2, Ngon> &) -> nd_array<std::int64_t>;  \
  template auto k_rings<std::int64_t, double, 2, Ngon>(                        \
      const mesh<std::int64_t, double, 2, Ngon> &, std::int32_t, bool)         \
      -> offset_blocked_buffer<std::int64_t, std::int64_t>;                    \
  template auto neighborhoods<std::int64_t, double, 2, Ngon>(                  \
      const mesh<std::int64_t, double, 2, Ngon> &, double, bool)               \
      -> offset_blocked_buffer<std::int64_t, std::int64_t>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_MESH_ANALYSIS)

#define TF_CPP_INSTANTIATE_ORIENT_FACES(Ngon)                                  \
  template auto orient_faces_consistently<std::int64_t, double, 2, Ngon>(      \
      const mesh<std::int64_t, double, 2, Ngon> &)                             \
      -> tf::polygons_buffer<std::int64_t, double, 2, Ngon>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_ORIENT_FACES)

#undef TF_CPP_INSTANTIATE_ORIENT_FACES
#undef TF_CPP_INSTANTIATE_MESH_ANALYSIS

} // namespace tf::cpp

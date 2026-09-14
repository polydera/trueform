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

#include <cstdint>

namespace tf::cpp {

#define TF_CPP_INSTANTIATE_TYPED_CLEAN_AT(Ngon)                                \
  template auto cleaned_mesh<std::int32_t, float, 3, Ngon>(                    \
      const mesh<std::int32_t, float, 3, Ngon> &, float, bool, bool)           \
      -> tf::polygons_buffer<std::int32_t, float, 3, Ngon>;                    \
  template auto cleaned_mesh_with_maps<std::int32_t, float, 3, Ngon>(          \
      const mesh<std::int32_t, float, 3, Ngon> &, float, bool, bool)           \
      -> cleaned_mesh_result<std::int32_t, float, 3, Ngon>

TF_CPP_MATRIX_FOR_EACH_NGON(TF_CPP_INSTANTIATE_TYPED_CLEAN_AT)

template auto cleaned_edge_mesh<std::int32_t, float, 3>(
    const edge_mesh<std::int32_t, float, 3> &, float, bool, bool)
    -> tf::segments_buffer<std::int32_t, float, 3>;
template auto cleaned_edge_mesh_with_maps<std::int32_t, float, 3>(
    const edge_mesh<std::int32_t, float, 3> &, float, bool, bool)
    -> cleaned_edge_mesh_result<std::int32_t, float, 3>;

#undef TF_CPP_INSTANTIATE_TYPED_CLEAN_AT

} // namespace tf::cpp

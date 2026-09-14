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
#pragma once

#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <type_traits>

namespace tf::cpp {
/// @brief Compute a fixed-triangle manifold-edge link, shaped `[F, 3]`.
template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto manifold_edge_link(
    const nd_array<Index> &faces,
    const offset_blocked_buffer<Index, Index> &face_membership)
    -> nd_array<Index>;

/// @brief Compute a dynamic manifold-edge link with offsets aligned to faces.
template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto manifold_edge_link(
    const offset_blocked_buffer<Index, Index> &faces,
    const offset_blocked_buffer<Index, Index> &face_membership)
    -> offset_blocked_buffer<Index, Index>;

#define TF_CPP_EXTERN_MANIFOLD_EDGE_LINK(Index)                                \
  extern template auto manifold_edge_link(                                     \
      const nd_array<Index> &, const offset_blocked_buffer<Index, Index> &)    \
      -> nd_array<Index>;                                                      \
  extern template auto manifold_edge_link(                                     \
      const offset_blocked_buffer<Index, Index> &,                             \
      const offset_blocked_buffer<Index, Index> &)                             \
      -> offset_blocked_buffer<Index, Index>

TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_EXTERN_MANIFOLD_EDGE_LINK)

#undef TF_CPP_EXTERN_MANIFOLD_EDGE_LINK

} // namespace tf::cpp

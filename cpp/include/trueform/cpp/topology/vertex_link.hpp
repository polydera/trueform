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

#include "trueform/cpp/core/detail/non_deduced.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <type_traits>

namespace tf::cpp {
/// @brief Compute bidirectional vertex adjacency from fixed edges.
template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto vertex_link_edges(const nd_array<Index> &edges,
                       detail::non_deduced_t<Index> n_ids)
    -> offset_blocked_buffer<Index, Index>;

/// @brief Compute vertex adjacency from fixed triangle faces and membership.
template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto vertex_link_faces(
    const nd_array<Index> &faces,
    const offset_blocked_buffer<Index, Index> &cell_membership)
    -> offset_blocked_buffer<Index, Index>;

/// @brief Compute vertex adjacency from dynamic faces and membership.
template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto vertex_link_faces(
    const offset_blocked_buffer<Index, Index> &faces,
    const offset_blocked_buffer<Index, Index> &cell_membership)
    -> offset_blocked_buffer<Index, Index>;

#define TF_CPP_EXTERN_VERTEX_LINK(Index)                                       \
  extern template auto vertex_link_edges(const nd_array<Index> &, Index)       \
      -> offset_blocked_buffer<Index, Index>;                                  \
  extern template auto vertex_link_faces(                                      \
      const nd_array<Index> &, const offset_blocked_buffer<Index, Index> &)    \
      -> offset_blocked_buffer<Index, Index>;                                  \
  extern template auto vertex_link_faces(                                      \
      const offset_blocked_buffer<Index, Index> &,                             \
      const offset_blocked_buffer<Index, Index> &)                             \
      -> offset_blocked_buffer<Index, Index>

TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_EXTERN_VERTEX_LINK)

#undef TF_CPP_EXTERN_VERTEX_LINK

} // namespace tf::cpp

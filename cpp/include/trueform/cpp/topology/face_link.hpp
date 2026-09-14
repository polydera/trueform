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
/// @brief Map each face to the faces it shares an edge with.
template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto face_link(const nd_array<Index> &faces,
               const offset_blocked_buffer<Index, Index> &face_membership)
    -> offset_blocked_buffer<Index, Index>;

/// @overload
template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto face_link(const offset_blocked_buffer<Index, Index> &faces,
               const offset_blocked_buffer<Index, Index> &face_membership)
    -> offset_blocked_buffer<Index, Index>;

#define TF_CPP_EXTERN_FACE_LINK(Index)                                         \
  extern template auto face_link(const nd_array<Index> &,                      \
                                 const offset_blocked_buffer<Index, Index> &)  \
      -> offset_blocked_buffer<Index, Index>;                                  \
  extern template auto face_link(const offset_blocked_buffer<Index, Index> &,  \
                                 const offset_blocked_buffer<Index, Index> &)  \
      -> offset_blocked_buffer<Index, Index>

TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_EXTERN_FACE_LINK)

#undef TF_CPP_EXTERN_FACE_LINK

} // namespace tf::cpp

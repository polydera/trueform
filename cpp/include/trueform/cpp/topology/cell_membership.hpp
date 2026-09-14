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
/// @brief Map each vertex ID to the fixed edge or triangle cells containing it.
template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto cell_membership(const nd_array<Index> &cells,
                     detail::non_deduced_t<Index> n_ids)
    -> offset_blocked_buffer<Index, Index>;

/// @brief Map each vertex ID to the dynamic cells containing it.
template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto cell_membership(const offset_blocked_buffer<Index, Index> &cells,
                     detail::non_deduced_t<Index> n_ids)
    -> offset_blocked_buffer<Index, Index>;

#define TF_CPP_EXTERN_CELL_MEMBERSHIP(Index)                                   \
  extern template auto cell_membership(const nd_array<Index> &, Index)         \
      -> offset_blocked_buffer<Index, Index>;                                  \
  extern template auto cell_membership(                                        \
      const offset_blocked_buffer<Index, Index> &, Index)                      \
      -> offset_blocked_buffer<Index, Index>

TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_EXTERN_CELL_MEMBERSHIP)

#undef TF_CPP_EXTERN_CELL_MEMBERSHIP

} // namespace tf::cpp

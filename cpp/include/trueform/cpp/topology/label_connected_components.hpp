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

#include <cstdint>
#include <type_traits>

namespace tf::cpp {
/// @brief The component each vertex belongs to, and how many there are.
template <typename Index = default_index_t> struct connected_components_result {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "connected_components_result requires an unqualified "
                "supported index type");

  nd_array<Index> labels;
  Index n_components;
};

/// The carriers name no index of their own, so the ENTRY is what states which
/// widths this build answers for: an index the matrix left out has no overload,
/// and naming it fails where it is called.
#define TF_CPP_DECLARE_LABEL_CONNECTED_COMPONENTS(Index)                       \
  auto label_connected_components(                                             \
      const offset_blocked_buffer<Index, Index> &connectivity)                 \
      ->connected_components_result<Index>;                                    \
  auto label_connected_components(                                             \
      const offset_blocked_buffer<Index, Index> &connectivity,                 \
      Index expected_number_of_components)                                     \
      ->connected_components_result<Index>;                                    \
  auto label_connected_components(const nd_array<Index> &connectivity)         \
      ->connected_components_result<Index>;                                    \
  auto label_connected_components(const nd_array<Index> &connectivity,         \
                                  Index expected_number_of_components)         \
      ->connected_components_result<Index>

TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_DECLARE_LABEL_CONNECTED_COMPONENTS)

#undef TF_CPP_DECLARE_LABEL_CONNECTED_COMPONENTS

} // namespace tf::cpp

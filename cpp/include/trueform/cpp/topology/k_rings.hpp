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
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tf::cpp {
/// @brief The vertices reachable within k connectivity hops.
/// @note Reads the VERTEX LINK.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto k_rings(const mesh<Index, Real, Dims, Ngon> &value, std::int32_t k,
             bool inclusive = false) -> offset_blocked_buffer<Index, Index>;

/// @brief The same, over a connectivity a caller already holds.
template <typename Index,
          std::enable_if_t<is_supported_index_v<Index>, int> = 0>
auto k_rings(const offset_blocked_buffer<Index, Index> &connectivity,
             std::int32_t k, bool inclusive = false)
    -> offset_blocked_buffer<Index, Index>;

#define TF_CPP_EXTERN_MESH_K_RINGS(Index, Real, Dims, Ngon)                    \
  extern template auto k_rings<Index, Real, Dims, Ngon>(                       \
      const mesh<Index, Real, Dims, Ngon> &, std::int32_t, bool)               \
      -> offset_blocked_buffer<Index, Index>

#define TF_CPP_EXTERN_K_RINGS(Index)                                           \
  extern template auto k_rings(const offset_blocked_buffer<Index, Index> &,    \
                               std::int32_t, bool)                             \
      -> offset_blocked_buffer<Index, Index>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_MESH_K_RINGS)
TF_CPP_MATRIX_FOR_EACH_INDEX(TF_CPP_EXTERN_K_RINGS)

#undef TF_CPP_EXTERN_K_RINGS
#undef TF_CPP_EXTERN_MESH_K_RINGS

} // namespace tf::cpp

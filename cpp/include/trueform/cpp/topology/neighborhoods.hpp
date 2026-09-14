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
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {
/// @brief The vertices reachable within a metric radius of each vertex.
/// @note Reads the VERTEX LINK.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto neighborhoods(const mesh<Index, Real, Dims, Ngon> &value, Real radius,
                   bool inclusive = false)
    -> offset_blocked_buffer<Index, Index>;

/// @brief The same, over a connectivity and points a caller already holds.
///
/// Dims is explicit because nd_array stores dimensionality at runtime. Points
/// must have shape `[connectivity.size(), Dims]`.
template <typename Index, typename Real, std::size_t Dims,
          std::enable_if_t<
              is_supported_index_v<Index> && (Dims == 2 || Dims == 3), int> = 0>
auto neighborhoods(const offset_blocked_buffer<Index, Index> &connectivity,
                   const nd_array<Real> &points, Real radius,
                   bool inclusive = false)
    -> offset_blocked_buffer<Index, Index>;

#define TF_CPP_EXTERN_MESH_NEIGHBORHOODS(Index, Real, Dims, Ngon)              \
  extern template auto neighborhoods<Index, Real, Dims, Ngon>(                 \
      const mesh<Index, Real, Dims, Ngon> &, Real, bool)                       \
      -> offset_blocked_buffer<Index, Index>

#define TF_CPP_EXTERN_NEIGHBORHOODS(Index, Real, Dims)                         \
  extern template auto neighborhoods<Index, Real, Dims>(                       \
      const offset_blocked_buffer<Index, Index> &, const nd_array<Real> &,     \
      Real, bool) -> offset_blocked_buffer<Index, Index>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_MESH_NEIGHBORHOODS)
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS(TF_CPP_EXTERN_NEIGHBORHOODS)

#undef TF_CPP_EXTERN_NEIGHBORHOODS
#undef TF_CPP_EXTERN_MESH_NEIGHBORHOODS

} // namespace tf::cpp

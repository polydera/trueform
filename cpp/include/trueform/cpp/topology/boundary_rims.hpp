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
#pragma once

#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace tf::cpp {
/// @brief A mesh's boundary rims: the vertices of each, the face across each
/// of its edges, and whether it closes.
///
/// Block `i` of `vertices` is rim `i` walked and block `i` of `faces` names the
/// face carrying each of its edges, so rim edge `k` runs from vertex `k` to
/// vertex `k + 1` and is carried by face `k` alone. A closed rim of `n`
/// vertices has `n` edges, the last running back to vertex `0`; an open one has
/// `n - 1`.
template <typename Index> struct boundary_rims_result {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "boundary_rims_result requires an unqualified supported index "
                "type");

  offset_blocked_buffer<Index, Index> vertices;
  offset_blocked_buffer<Index, Index> faces;
  /// @brief Shaped [R]: nonzero where rim `i`'s last edge runs back to its
  /// first vertex.
  nd_array<std::int8_t> closed;
};

/// @brief Assemble a mesh's boundary edges into rims.
/// @note Reads the FACE MEMBERSHIP.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto boundary_rims(const mesh<Index, Real, Dims, Ngon> &value)
    -> boundary_rims_result<Index>;

#define TF_CPP_EXTERN_BOUNDARY_RIMS(Index, Real, Dims, Ngon)                   \
  extern template auto boundary_rims<Index, Real, Dims, Ngon>(                 \
      const mesh<Index, Real, Dims, Ngon> &) -> boundary_rims_result<Index>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_BOUNDARY_RIMS)

#undef TF_CPP_EXTERN_BOUNDARY_RIMS

} // namespace tf::cpp

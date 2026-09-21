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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/index_type.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {
/// @brief The separated geometry and the input point each of its points copies.
template <typename Index, typename Real, std::size_t Dims = 3,
          std::size_t Ngon = 3>
struct split_non_manifold_vertices_result {
  static_assert(is_supported_index_v<Index> &&
                    std::is_same_v<Index, std::remove_cv_t<Index>>,
                "split_non_manifold_vertices_result requires an unqualified "
                "supported index type");
  static_assert(Dims == 2 || Dims == 3,
                "split_non_manifold_vertices_result Dims must be 2 or 3");

  tf::polygons_buffer<Index, Real, Dims, Ngon> mesh;
  /// @brief Shaped [P]: for each output point the input point it copies.
  nd_array<Index> point_map;
};

/// @brief The same geometry, in storage of its own, with every fan at a vertex
/// given a vertex of its own.
///
/// Faces keep their ids, their arity and their winding, and the points are the
/// input's followed by the minted copies. The frame is the reading's and does
/// not travel with the coordinates. A vertex any of whose edges carries 3+
/// faces is left untouched — separating its fans would tear that edge into
/// boundary copies — and `non_manifold_vertices` still names it.
/// @note Reads the FACE MEMBERSHIP.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto split_non_manifold_vertices(const mesh<Index, Real, Dims, Ngon> &value)
    -> split_non_manifold_vertices_result<Index, Real, Dims, Ngon>;

#define TF_CPP_EXTERN_SPLIT_NON_MANIFOLD_VERTICES(Index, Real, Dims, Ngon)     \
  extern template auto split_non_manifold_vertices<Index, Real, Dims, Ngon>(   \
      const mesh<Index, Real, Dims, Ngon> &)                                   \
      -> split_non_manifold_vertices_result<Index, Real, Dims, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(
    TF_CPP_EXTERN_SPLIT_NON_MANIFOLD_VERTICES)

#undef TF_CPP_EXTERN_SPLIT_NON_MANIFOLD_VERTICES

} // namespace tf::cpp

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

#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"

#include <cstddef>

namespace tf::cpp {
/// @brief The vertices whose faces are not one fan, ascending, shaped [N].
/// @note Reads the FACE MEMBERSHIP.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto non_manifold_vertices(const mesh<Index, Real, Dims, Ngon> &value)
    -> nd_array<Index>;

#define TF_CPP_EXTERN_NON_MANIFOLD_VERTICES(Index, Real, Dims, Ngon)           \
  extern template auto non_manifold_vertices<Index, Real, Dims, Ngon>(         \
      const mesh<Index, Real, Dims, Ngon> &) -> nd_array<Index>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_NON_MANIFOLD_VERTICES)

#undef TF_CPP_EXTERN_NON_MANIFOLD_VERTICES

} // namespace tf::cpp

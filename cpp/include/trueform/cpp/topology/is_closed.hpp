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

#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"

#include <cstddef>

namespace tf::cpp {
/// @note Reads the FACE MEMBERSHIP.
/// @brief Whether every edge of the mesh is shared by two faces.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto is_closed(const mesh<Index, Real, Dims, Ngon> &value) -> bool;

/// @brief Whether the mesh has a boundary.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto is_open(const mesh<Index, Real, Dims, Ngon> &value) -> bool;

#define TF_CPP_EXTERN_IS_CLOSED(Index, Real, Dims, Ngon)                       \
  extern template auto is_closed<Index, Real, Dims, Ngon>(                     \
      const mesh<Index, Real, Dims, Ngon> &) -> bool;                          \
  extern template auto is_open<Index, Real, Dims, Ngon>(                       \
      const mesh<Index, Real, Dims, Ngon> &) -> bool

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_IS_CLOSED)

#undef TF_CPP_EXTERN_IS_CLOSED

} // namespace tf::cpp

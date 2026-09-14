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

/// @brief Compute the maximum polygon-edge length in the transformed frame.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto max_edge_length(const mesh<Index, Real, Dims, Ngon> &value) -> Real;

#define TF_CPP_EXTERN_MESH_MAX_EDGE_LENGTH(Index, Real, Dims, Ngon)            \
  extern template auto max_edge_length<Index, Real, Dims, Ngon>(               \
      const mesh<Index, Real, Dims, Ngon> &) -> Real

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_MESH_MAX_EDGE_LENGTH)

#undef TF_CPP_EXTERN_MESH_MAX_EDGE_LENGTH

} // namespace tf::cpp

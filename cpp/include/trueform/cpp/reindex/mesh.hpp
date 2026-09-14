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

#include "trueform/core/polygons_buffer.hpp"
#include "trueform/cpp/core/index_map.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Apply typed face and point maps to mesh connectivity.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reindexed(const mesh<Index, Real, Dims, Ngon> &value,
               const index_map<Index> &face_map,
               const index_map<Index> &point_map)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon>;

#define TF_CPP_REINDEX_EXTERN_MESH(Index, Real, Dims, Ngon)                    \
  extern template auto reindexed<Index, Real, Dims, Ngon>(                     \
      const mesh<Index, Real, Dims, Ngon> &, const index_map<Index> &,         \
      const index_map<Index> &)                                                \
      -> tf::polygons_buffer<Index, Real, Dims, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_REINDEX_EXTERN_MESH)

#undef TF_CPP_REINDEX_EXTERN_MESH

} // namespace tf::cpp

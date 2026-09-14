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

#include "trueform/core/segments_buffer.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/index_map.hpp"
#include "trueform/cpp/core/matrix.hpp"

#include <cstddef>

namespace tf::cpp {

/// @brief Owning result of edge-mesh cleaning with source-to-result maps.
template <typename Index, typename Real, std::size_t Dims = 3>
struct cleaned_edge_mesh_result {
  tf::segments_buffer<Index, Real, Dims> edge_mesh;
  index_map<Index> edge_map;
  index_map<Index> point_map;
};

/// @brief Clean a typed 2D or 3D edge mesh in local coordinates.
template <typename Index, typename Real, std::size_t Dims>
auto cleaned_edge_mesh(const edge_mesh<Index, Real, Dims> &value,
                       Real tolerance = Real{},
                       bool remove_duplicate_primitives = true,
                       bool remove_unreferenced_points = true)
    -> tf::segments_buffer<Index, Real, Dims>;

/// @brief Clean a typed edge mesh and return maps in its index type.
template <typename Index, typename Real, std::size_t Dims>
auto cleaned_edge_mesh_with_maps(const edge_mesh<Index, Real, Dims> &value,
                                 Real tolerance = Real{},
                                 bool remove_duplicate_primitives = true,
                                 bool remove_unreferenced_points = true)
    -> cleaned_edge_mesh_result<Index, Real, Dims>;

#define TF_CPP_EXTERN_CLEAN_EDGE_MESH(Index, Real, Dims)                       \
  extern template auto cleaned_edge_mesh<Index, Real, Dims>(                   \
      const edge_mesh<Index, Real, Dims> &, Real, bool, bool)                  \
      -> tf::segments_buffer<Index, Real, Dims>;                               \
  extern template auto cleaned_edge_mesh_with_maps<Index, Real, Dims>(         \
      const edge_mesh<Index, Real, Dims> &, Real, bool, bool)                  \
      -> cleaned_edge_mesh_result<Index, Real, Dims>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS(TF_CPP_EXTERN_CLEAN_EDGE_MESH)

#undef TF_CPP_EXTERN_CLEAN_EDGE_MESH

} // namespace tf::cpp

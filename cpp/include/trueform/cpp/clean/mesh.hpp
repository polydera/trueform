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

/// @brief Owning result of mesh cleaning with source-to-result maps.
template <typename Index, typename Real, std::size_t Dims = 3,
          std::size_t Ngon = 3>
struct cleaned_mesh_result {
  tf::polygons_buffer<Index, Real, Dims, Ngon> mesh;
  index_map<Index> face_map;
  index_map<Index> point_map;
};

/// @brief Clean a 2D or 3D fixed/dynamic mesh in local coordinates.
///
/// The result is core's own storage in the coordinates the operand was
/// authored in, and preserves the input connectivity layout and index width.
/// Zero selects exact cleaning; negative and nonfinite tolerances are rejected.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cleaned_mesh(const mesh<Index, Real, Dims, Ngon> &value,
                  Real tolerance = Real{},
                  bool remove_duplicate_primitives = true,
                  bool remove_unreferenced_points = true)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon>;

/// @brief Clean a typed mesh and return maps in the mesh's index type.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto cleaned_mesh_with_maps(const mesh<Index, Real, Dims, Ngon> &value,
                            Real tolerance = Real{},
                            bool remove_duplicate_primitives = true,
                            bool remove_unreferenced_points = true)
    -> cleaned_mesh_result<Index, Real, Dims, Ngon>;

#define TF_CPP_EXTERN_CLEAN_MESH(Index, Real, Dims, Ngon)                      \
  extern template auto cleaned_mesh<Index, Real, Dims, Ngon>(                  \
      const mesh<Index, Real, Dims, Ngon> &, Real, bool, bool)                 \
      -> tf::polygons_buffer<Index, Real, Dims, Ngon>;                         \
  extern template auto cleaned_mesh_with_maps<Index, Real, Dims, Ngon>(        \
      const mesh<Index, Real, Dims, Ngon> &, Real, bool, bool)                 \
      -> cleaned_mesh_result<Index, Real, Dims, Ngon>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_CLEAN_MESH)

#undef TF_CPP_EXTERN_CLEAN_MESH

} // namespace tf::cpp

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
#include "trueform/core/segments_buffer.hpp"
#include "trueform/cpp/core/common_index.hpp"
#include "trueform/cpp/core/detail/concatenated_arity.hpp"
#include "trueform/cpp/core/detail/face_blocks_array.hpp"
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/reindex/detail/array_mesh.hpp"

#include <cstddef>
#include <type_traits>
#include <vector>

namespace tf::cpp {

/// @brief Concatenate homogeneous meshes, applying every placement.
///
/// A range is homogeneous by definition, so its element's arity is the
/// result's.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto concatenate_meshes(
    const std::vector<mesh<Index, Real, Dims, Ngon>> &meshes)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon>;

/// @brief Concatenate homogeneous edge meshes, applying every placement.
template <typename Index, typename Real, std::size_t Dims>
auto concatenate_edge_meshes(
    const std::vector<edge_mesh<Index, Real, Dims>> &meshes)
    -> tf::segments_buffer<Index, Real, Dims>;

/// @brief Concatenate a heterogeneous mesh pair. Coordinates and indices use
/// their respective common types; mixed fixed/dynamic connectivity becomes
/// dynamic, and both placements are applied before promotion.
template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims, std::size_t Ngon0, std::size_t Ngon1>
auto concatenate_meshes(const mesh<Index0, Real0, Dims, Ngon0> &first,
                        const mesh<Index1, Real1, Dims, Ngon1> &second)
    -> tf::polygons_buffer<common_index_t<Index0, Index1>,
                           std::common_type_t<Real0, Real1>, Dims,
                           detail::concatenated_arity_v<Ngon0, Ngon1>>;

/// @brief Concatenate a heterogeneous edge-mesh pair with common-type
/// coordinate and index storage, applying both placements.
template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims>
auto concatenate_edge_meshes(const edge_mesh<Index0, Real0, Dims> &first,
                             const edge_mesh<Index1, Real1, Dims> &second)
    -> tf::segments_buffer<common_index_t<Index0, Index1>,
                           std::common_type_t<Real0, Real1>, Dims>;

/// @brief Concatenate heterogeneous tuple-equivalent mesh arrays. Each
/// connectivity may independently state fixed triangles or dynamic faces.
/// Every supported dtype pair delegates to the explicitly instantiated
/// heterogeneous mesh pair, so this entry compiles no kernel of its own.
template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims, std::size_t Ngon0, std::size_t Ngon1>
auto concatenate_meshes(
    const detail::face_blocks_array_t<Index0, Ngon0> &first_faces,
    const nd_array<Real0> &first_points,
    const detail::face_blocks_array_t<Index1, Ngon1> &second_faces,
    const nd_array<Real1> &second_points)
    -> tf::polygons_buffer<common_index_t<Index0, Index1>,
                           std::common_type_t<Real0, Real1>, Dims,
                           detail::concatenated_arity_v<Ngon0, Ngon1>> {
  detail::array_mesh<Index0, Real0, Dims, Ngon0> first(first_faces,
                                                       first_points);
  detail::array_mesh<Index1, Real1, Dims, Ngon1> second(second_faces,
                                                        second_points);
  return concatenate_meshes(first.mesh(), second.mesh());
}

/// @brief Concatenate heterogeneous tuple-equivalent edge arrays.
template <typename Index0, typename Real0, typename Index1, typename Real1,
          std::size_t Dims>
auto concatenate_edge_meshes(const nd_array<Index0> &first_edges,
                             const nd_array<Real0> &first_points,
                             const nd_array<Index1> &second_edges,
                             const nd_array<Real1> &second_points)
    -> tf::segments_buffer<common_index_t<Index0, Index1>,
                           std::common_type_t<Real0, Real1>, Dims> {
  detail::array_edge_mesh<Index0, Real0, Dims> first(first_edges, first_points);
  detail::array_edge_mesh<Index1, Real1, Dims> second(second_edges,
                                                      second_points);
  return concatenate_edge_meshes(first.edge_mesh(), second.edge_mesh());
}

#define TF_CPP_REINDEX_EXTERN_CONCATENATE_MESHES(Index, Real, Dims, Ngon)      \
  extern template auto concatenate_meshes<Index, Real, Dims, Ngon>(            \
      const std::vector<mesh<Index, Real, Dims, Ngon>> &)                      \
      -> tf::polygons_buffer<Index, Real, Dims, Ngon>

#define TF_CPP_REINDEX_EXTERN_CONCATENATE_EDGE_MESHES(Index, Real, Dims)       \
  extern template auto concatenate_edge_meshes<Index, Real, Dims>(             \
      const std::vector<edge_mesh<Index, Real, Dims>> &)                       \
      -> tf::segments_buffer<Index, Real, Dims>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(
    TF_CPP_REINDEX_EXTERN_CONCATENATE_MESHES)
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS(
    TF_CPP_REINDEX_EXTERN_CONCATENATE_EDGE_MESHES)

#undef TF_CPP_REINDEX_EXTERN_CONCATENATE_EDGE_MESHES
#undef TF_CPP_REINDEX_EXTERN_CONCATENATE_MESHES

/// A concatenation reads TWO operands, so its declarations cross both operands'
/// index, real and arity axes at the dimension they share -- which is what the
/// shards instantiate, and the one shape in the layer that needs all seven.
#define TF_CPP_REINDEX_EXTERN_MESH_PAIR(Index0, Real0, Dims, Index1, Real1,    \
                                        Ngon0, Ngon1)                          \
  extern template auto                                                         \
  concatenate_meshes<Index0, Real0, Index1, Real1, Dims, Ngon0, Ngon1>(        \
      const mesh<Index0, Real0, Dims, Ngon0> &,                                \
      const mesh<Index1, Real1, Dims, Ngon1> &)                                \
      -> tf::polygons_buffer<common_index_t<Index0, Index1>,                   \
                             std::common_type_t<Real0, Real1>, Dims,           \
                             detail::concatenated_arity_v<Ngon0, Ngon1>>

#define TF_CPP_REINDEX_EXTERN_EDGE_MESH_PAIR(Index0, Real0, Dims, Index1,      \
                                             Real1)                            \
  extern template auto                                                         \
  concatenate_edge_meshes<Index0, Real0, Index1, Real1, Dims>(                 \
      const edge_mesh<Index0, Real0, Dims> &,                                  \
      const edge_mesh<Index1, Real1, Dims> &)                                  \
      -> tf::segments_buffer<common_index_t<Index0, Index1>,                   \
                             std::common_type_t<Real0, Real1>, Dims>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_INDEX_REAL_NGON_PAIR(
    TF_CPP_REINDEX_EXTERN_MESH_PAIR)
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_INDEX_REAL(
    TF_CPP_REINDEX_EXTERN_EDGE_MESH_PAIR)

#undef TF_CPP_REINDEX_EXTERN_EDGE_MESH_PAIR
#undef TF_CPP_REINDEX_EXTERN_MESH_PAIR

} // namespace tf::cpp

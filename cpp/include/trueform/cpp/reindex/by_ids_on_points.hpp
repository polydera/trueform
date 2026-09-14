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
#include "trueform/cpp/core/edge_mesh.hpp"
#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"
#include "trueform/cpp/core/nd_array.hpp"
#include "trueform/cpp/core/offset_blocked_buffer.hpp"
#include "trueform/cpp/reindex/fixed_arity.hpp"
#include "trueform/cpp/reindex/selection_result.hpp"

#include <cstddef>
#include <type_traits>

namespace tf::cpp {

/// @brief Select mesh points by IDs, keeping the carrier's index width.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reindexed_by_ids_on_points(const mesh<Index, Real, Dims, Ngon> &value,
                                const nd_array<Index> &ids)
    -> tf::polygons_buffer<Index, Real, Dims, Ngon>;

template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto reindexed_by_ids_on_points_with_maps(
    const mesh<Index, Real, Dims, Ngon> &value, const nd_array<Index> &ids)
    -> reindexed_mesh_result<Index, Real, Dims, Ngon>;


/// @brief Select typed edge-mesh points by IDs.
template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_ids_on_points(
    const edge_mesh<Index, Real, Dims> &value,
    const nd_array<Index> &ids) -> tf::segments_buffer<Index, Real, Dims>;

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_ids_on_points_with_maps(
    const edge_mesh<Index, Real, Dims> &value,
    const nd_array<Index> &ids)
    -> reindexed_edge_mesh_result<Index, Real, Dims>;

/// Tuple-equivalent fixed and dynamic mesh inputs, validated into a public
/// mesh carrier so one typed computation answers every entry shape.
template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_ids_on_points(const nd_array<Index> &faces,
                                const nd_array<Real> &points,
                                const nd_array<Index> &ids)
    -> tf::polygons_buffer<Index, Real, Dims, 3>;

template <typename Index, typename Real, std::size_t Dims>
auto reindexed_by_ids_on_points(
    const offset_blocked_buffer<Index, Index> &faces,
    const nd_array<Real> &points, const nd_array<Index> &ids)
    -> tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size>;

/// Fixed tuple-equivalent arrays with arity encoded as 2 (edges) or 3
/// (triangles). Explicit arity avoids a runtime variant return type.
template <typename Index, typename Real, std::size_t Dims, std::size_t Vertices,
          std::enable_if_t<Vertices == 2 || Vertices == 3, int> = 0>
auto reindexed_by_ids_on_points(const nd_array<Index> &elements,
                                const nd_array<Real> &points,
                                const nd_array<Index> &ids)
    -> reindexed_fixed_carrier_t<Index, Real, Dims, Vertices>;

#define TF_CPP_REINDEX_EXTERN_BY_IDS_ON_POINTS_MESH(Index, Real, Dims, Ngon)   \
  extern template auto reindexed_by_ids_on_points<Index, Real, Dims, Ngon>(    \
      const mesh<Index, Real, Dims, Ngon> &, const nd_array<Index> &)          \
      -> tf::polygons_buffer<Index, Real, Dims, Ngon>;                                 \
  extern template auto                                                         \
  reindexed_by_ids_on_points_with_maps<Index, Real, Dims, Ngon>(               \
      const mesh<Index, Real, Dims, Ngon> &, const nd_array<Index> &)          \
      -> reindexed_mesh_result<Index, Real, Dims, Ngon>

#define TF_CPP_REINDEX_EXTERN_BY_IDS_ON_POINTS(Index, Real, Dims)              \
  extern template auto reindexed_by_ids_on_points<Index, Real, Dims>(          \
      const edge_mesh<Index, Real, Dims> &, const nd_array<Index> &)    \
      -> tf::segments_buffer<Index, Real, Dims>;                                  \
  extern template auto                                                         \
  reindexed_by_ids_on_points_with_maps<Index, Real, Dims>(                     \
      const edge_mesh<Index, Real, Dims> &, const nd_array<Index> &)    \
      -> reindexed_edge_mesh_result<Index, Real, Dims>;                        \
  extern template auto reindexed_by_ids_on_points<Index, Real, Dims, 2>(       \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<Index> &)                                                 \
      -> reindexed_fixed_carrier_t<Index, Real, Dims, 2>;                      \
  extern template auto reindexed_by_ids_on_points<Index, Real, Dims, 3>(       \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<Index> &)                                                 \
      -> reindexed_fixed_carrier_t<Index, Real, Dims, 3>

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(
    TF_CPP_REINDEX_EXTERN_BY_IDS_ON_POINTS_MESH)
TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS(TF_CPP_REINDEX_EXTERN_BY_IDS_ON_POINTS)

#undef TF_CPP_REINDEX_EXTERN_BY_IDS_ON_POINTS
#undef TF_CPP_REINDEX_EXTERN_BY_IDS_ON_POINTS_MESH

} // namespace tf::cpp

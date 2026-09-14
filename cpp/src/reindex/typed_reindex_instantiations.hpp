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
/* Explicit-instantiation matrix for one Real x Index x Dims reindex shard. */
#pragma once

#include "split_components_impl.hpp"
#include "typed_reindex_impl.hpp"

#include <cstdint>
#include <type_traits>
#include <vector>

#define TF_CPP_INSTANTIATE_HETEROGENEOUS_PAIR_AT(Real0, Index0, Real1, Index1, \
                                                 Dims, Ngon0, Ngon1)           \
  template auto                                                                \
  concatenate_meshes<Index0, Real0, Index1, Real1, Dims, Ngon0, Ngon1>(        \
      const mesh<Index0, Real0, Dims, Ngon0> &,                                \
      const mesh<Index1, Real1, Dims, Ngon1> &)                                \
      -> tf::polygons_buffer<common_index_t<Index0, Index1>,                   \
                             std::common_type_t<Real0, Real1>, Dims,           \
                             detail::concatenated_arity_v<Ngon0, Ngon1>>

#define TF_CPP_INSTANTIATE_HETEROGENEOUS_EDGE_PAIR(Real0, Index0, Real1,       \
                                                   Index1, Dims)               \
  template auto concatenate_edge_meshes<Index0, Real0, Index1, Real1, Dims>(   \
      const edge_mesh<Index0, Real0, Dims> &,                                  \
      const edge_mesh<Index1, Real1, Dims> &)                                  \
      -> tf::segments_buffer<common_index_t<Index0, Index1>,                   \
                             std::common_type_t<Real0, Real1>, Dims>

#define TF_CPP_INSTANTIATE_MESH_REINDEX(Real, Index, Dims, Ngon)               \
  template auto reindexed<Index, Real, Dims, Ngon>(                            \
      const mesh<Index, Real, Dims, Ngon> &, const index_map<Index> &,         \
      const index_map<Index> &)                                                \
      -> tf::polygons_buffer<Index, Real, Dims, Ngon>;                         \
  template auto reindexed_by_ids<Index, Real, Dims, Ngon>(                     \
      const mesh<Index, Real, Dims, Ngon> &, const nd_array<Index> &)          \
      -> tf::polygons_buffer<Index, Real, Dims, Ngon>;                         \
  template auto reindexed_by_ids_with_maps<Index, Real, Dims, Ngon>(           \
      const mesh<Index, Real, Dims, Ngon> &, const nd_array<Index> &)          \
      -> reindexed_mesh_result<Index, Real, Dims, Ngon>;                       \
  template auto reindexed_by_mask<Index, Real, Dims, Ngon>(                    \
      const mesh<Index, Real, Dims, Ngon> &, const nd_array<std::int8_t> &)    \
      -> tf::polygons_buffer<Index, Real, Dims, Ngon>;                         \
  template auto reindexed_by_mask_with_maps<Index, Real, Dims, Ngon>(          \
      const mesh<Index, Real, Dims, Ngon> &, const nd_array<std::int8_t> &)    \
      -> reindexed_mesh_result<Index, Real, Dims, Ngon>;                       \
  template auto reindexed_by_ids_on_points<Index, Real, Dims, Ngon>(           \
      const mesh<Index, Real, Dims, Ngon> &, const nd_array<Index> &)          \
      -> tf::polygons_buffer<Index, Real, Dims, Ngon>;                         \
  template auto reindexed_by_ids_on_points_with_maps<Index, Real, Dims, Ngon>( \
      const mesh<Index, Real, Dims, Ngon> &, const nd_array<Index> &)          \
      -> reindexed_mesh_result<Index, Real, Dims, Ngon>;                       \
  template auto reindexed_by_mask_on_points<Index, Real, Dims, Ngon>(          \
      const mesh<Index, Real, Dims, Ngon> &, const nd_array<std::int8_t> &)    \
      -> tf::polygons_buffer<Index, Real, Dims, Ngon>;                         \
  template auto                                                                \
  reindexed_by_mask_on_points_with_maps<Index, Real, Dims, Ngon>(              \
      const mesh<Index, Real, Dims, Ngon> &, const nd_array<std::int8_t> &)    \
      -> reindexed_mesh_result<Index, Real, Dims, Ngon>;                       \
  template auto split_into_components<Index, Real, Dims, Ngon>(                \
      const mesh<Index, Real, Dims, Ngon> &,                                   \
      const nd_array<std::int32_t> &)                                          \
      -> split_components_result<                                              \
          tf::polygons_buffer<Index, Real, Dims, Ngon>>;                       \
  template auto concatenate_meshes<Index, Real, Dims, Ngon>(                   \
      const std::vector<mesh<Index, Real, Dims, Ngon>> &)                      \
      -> tf::polygons_buffer<Index, Real, Dims, Ngon>;                         \
  template auto split_into_domains<Index, Real, Dims, Ngon>(                   \
      const mesh<Index, Real, Dims, Ngon> &,                                   \
      const domain_labels_result<Index> &)                                     \
      -> split_domains_result<Index, Real, Dims, Ngon>

#define TF_CPP_INSTANTIATE_TYPED_REINDEX(Real, Index, Dims)                    \
  template auto split_into_components<Index, Real, Dims>(                      \
      const edge_mesh<Index, Real, Dims> &,                                    \
      const nd_array<std::int32_t> &)                                          \
      -> split_components_result<tf::segments_buffer<Index, Real, Dims>>;      \
  template auto reindexed_by_ids<Index, Real, Dims>(                           \
      const nd_array<Real> &, const nd_array<Index> &) -> nd_array<Real>;      \
  template auto reindexed_by_ids_with_maps<Index, Real, Dims>(                 \
      const nd_array<Real> &, const nd_array<Index> &)                         \
      -> reindexed_points_result<Index, Real>;                                 \
  template auto reindexed_by_mask<Index, Real, Dims>(                          \
      const nd_array<Real> &, const nd_array<std::int8_t> &)                   \
      -> nd_array<Real>;                                                       \
  template auto reindexed_by_mask_with_maps<Index, Real, Dims>(                \
      const nd_array<Real> &, const nd_array<std::int8_t> &)                   \
      -> reindexed_points_result<Index, Real>;                                 \
  template auto reindexed_by_ids<Index, Real, Dims>(                           \
      const edge_mesh<Index, Real, Dims> &, const nd_array<Index> &)           \
      -> tf::segments_buffer<Index, Real, Dims>;                               \
  template auto reindexed_by_ids_with_maps<Index, Real, Dims>(                 \
      const edge_mesh<Index, Real, Dims> &, const nd_array<Index> &)           \
      -> reindexed_edge_mesh_result<Index, Real, Dims>;                        \
  template auto reindexed_by_mask<Index, Real, Dims>(                          \
      const edge_mesh<Index, Real, Dims> &, const nd_array<std::int8_t> &)     \
      -> tf::segments_buffer<Index, Real, Dims>;                               \
  template auto reindexed_by_mask_with_maps<Index, Real, Dims>(                \
      const edge_mesh<Index, Real, Dims> &, const nd_array<std::int8_t> &)     \
      -> reindexed_edge_mesh_result<Index, Real, Dims>;                        \
  template auto reindexed_by_ids_on_points<Index, Real, Dims>(                 \
      const edge_mesh<Index, Real, Dims> &, const nd_array<Index> &)           \
      -> tf::segments_buffer<Index, Real, Dims>;                               \
  template auto reindexed_by_ids_on_points_with_maps<Index, Real, Dims>(       \
      const edge_mesh<Index, Real, Dims> &, const nd_array<Index> &)           \
      -> reindexed_edge_mesh_result<Index, Real, Dims>;                        \
  template auto reindexed_by_mask_on_points<Index, Real, Dims>(                \
      const edge_mesh<Index, Real, Dims> &, const nd_array<std::int8_t> &)     \
      -> tf::segments_buffer<Index, Real, Dims>;                               \
  template auto reindexed_by_mask_on_points_with_maps<Index, Real, Dims>(      \
      const edge_mesh<Index, Real, Dims> &, const nd_array<std::int8_t> &)     \
      -> reindexed_edge_mesh_result<Index, Real, Dims>;                        \
  template auto reindexed_by_ids<Index, Real, Dims, 2>(                        \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<Index> &) -> tf::segments_buffer<Index, Real, Dims>;      \
  template auto reindexed_by_ids_with_maps<Index, Real, Dims, 2>(              \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<Index> &)                                                 \
      -> reindexed_edge_mesh_result<Index, Real, Dims>;                        \
  template auto reindexed_by_mask<Index, Real, Dims, 2>(                       \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<std::int8_t> &)                                           \
      -> tf::segments_buffer<Index, Real, Dims>;                               \
  template auto reindexed_by_mask_with_maps<Index, Real, Dims, 2>(             \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<std::int8_t> &)                                           \
      -> reindexed_edge_mesh_result<Index, Real, Dims>;                        \
  template auto reindexed_by_ids_on_points<Index, Real, Dims, 2>(              \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<Index> &) -> tf::segments_buffer<Index, Real, Dims>;      \
  template auto reindexed_by_mask_on_points<Index, Real, Dims, 2>(             \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<std::int8_t> &)                                           \
      -> tf::segments_buffer<Index, Real, Dims>;                               \
  template auto concatenate_edge_meshes<Index, Real, Dims>(                    \
      const std::vector<edge_mesh<Index, Real, Dims>> &)                       \
      -> tf::segments_buffer<Index, Real, Dims>

/// A soup states its own layout: fixed blocks are triangles, offset blocks are
/// mixed, so each family exists only where the matrix carries that layout.
#define TF_CPP_INSTANTIATE_TRIANGLE_SOUP_REINDEX(Real, Index, Dims)            \
  template auto reindexed_by_ids<Index, Real, Dims, 3>(                        \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<Index> &) -> tf::polygons_buffer<Index, Real, Dims, 3>;   \
  template auto reindexed_by_ids_with_maps<Index, Real, Dims, 3>(              \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<Index> &) -> reindexed_mesh_result<Index, Real, Dims>;    \
  template auto reindexed_by_mask<Index, Real, Dims, 3>(                       \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<std::int8_t> &)                                           \
      -> tf::polygons_buffer<Index, Real, Dims, 3>;                            \
  template auto reindexed_by_mask_with_maps<Index, Real, Dims, 3>(             \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<std::int8_t> &)                                           \
      -> reindexed_mesh_result<Index, Real, Dims>;                             \
  template auto reindexed_by_ids_on_points<Index, Real, Dims, 3>(              \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<Index> &) -> tf::polygons_buffer<Index, Real, Dims, 3>;   \
  template auto reindexed_by_mask_on_points<Index, Real, Dims, 3>(             \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<std::int8_t> &)                                           \
      -> tf::polygons_buffer<Index, Real, Dims, 3>;                            \
  template auto reindexed_by_ids<Index, Real, Dims>(                           \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<Index> &) -> tf::polygons_buffer<Index, Real, Dims, 3>;   \
  template auto reindexed_by_mask<Index, Real, Dims>(                          \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<std::int8_t> &)                                           \
      -> tf::polygons_buffer<Index, Real, Dims, 3>;                            \
  template auto reindexed_by_ids_with_maps<Index, Real, Dims>(                 \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<Index> &) -> reindexed_mesh_result<Index, Real, Dims>;    \
  template auto reindexed_by_mask_with_maps<Index, Real, Dims>(                \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<std::int8_t> &)                                           \
      -> reindexed_mesh_result<Index, Real, Dims>;                             \
  template auto reindexed_by_ids_on_points<Index, Real, Dims>(                 \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<Index> &) -> tf::polygons_buffer<Index, Real, Dims, 3>;   \
  template auto reindexed_by_mask_on_points<Index, Real, Dims>(                \
      const nd_array<Index> &, const nd_array<Real> &,                         \
      const nd_array<std::int8_t> &)                                           \
      -> tf::polygons_buffer<Index, Real, Dims, 3>

#define TF_CPP_INSTANTIATE_MIXED_SOUP_REINDEX(Real, Index, Dims)               \
  template auto reindexed_by_ids<Index, Real, Dims>(                           \
      const offset_blocked_buffer<Index, Index> &, const nd_array<Real> &,     \
      const nd_array<Index> &)                                                 \
      -> tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size>;             \
  template auto reindexed_by_mask<Index, Real, Dims>(                          \
      const offset_blocked_buffer<Index, Index> &, const nd_array<Real> &,     \
      const nd_array<std::int8_t> &)                                           \
      -> tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size>;             \
  template auto reindexed_by_ids_with_maps<Index, Real, Dims>(                 \
      const offset_blocked_buffer<Index, Index> &, const nd_array<Real> &,     \
      const nd_array<Index> &)                                                 \
      -> reindexed_mesh_result<Index, Real, Dims, tf::dynamic_size>;           \
  template auto reindexed_by_mask_with_maps<Index, Real, Dims>(                \
      const offset_blocked_buffer<Index, Index> &, const nd_array<Real> &,     \
      const nd_array<std::int8_t> &)                                           \
      -> reindexed_mesh_result<Index, Real, Dims, tf::dynamic_size>;           \
  template auto reindexed_by_ids_on_points<Index, Real, Dims>(                 \
      const offset_blocked_buffer<Index, Index> &, const nd_array<Real> &,     \
      const nd_array<Index> &)                                                 \
      -> tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size>;             \
  template auto reindexed_by_mask_on_points<Index, Real, Dims>(                \
      const offset_blocked_buffer<Index, Index> &, const nd_array<Real> &,     \
      const nd_array<std::int8_t> &)                                           \
      -> tf::polygons_buffer<Index, Real, Dims, tf::dynamic_size>

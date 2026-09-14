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

/// The connectivity entries an index's own shard defines. One text, invoked by
/// each shard with the index it is per, so a consumer that names one index
/// extracts one shard.
#define TF_CPP_DEFINE_TOPOLOGY_COMPONENTS(Index)                               \
  auto connect_edges_to_paths(const nd_array<Index> &edges)                    \
      -> offset_blocked_buffer<Index, Index> {                                 \
    return detail::connect_edges_to_paths_impl(edges);                         \
  }                                                                            \
                                                                               \
  auto label_connected_components(                                             \
      const offset_blocked_buffer<Index, Index> &connectivity)                 \
      -> connected_components_result<Index> {                                  \
    return detail::label_offset_blocked_connected_components(connectivity,     \
                                                             Index{2});        \
  }                                                                            \
                                                                               \
  auto label_connected_components(                                             \
      const offset_blocked_buffer<Index, Index> &connectivity,                 \
      Index expected_number_of_components)                                     \
      -> connected_components_result<Index> {                                  \
    return detail::label_offset_blocked_connected_components(                  \
        connectivity, expected_number_of_components);                          \
  }                                                                            \
                                                                               \
  auto label_connected_components(const nd_array<Index> &connectivity)         \
      -> connected_components_result<Index> {                                  \
    return detail::label_dense_connected_components(connectivity, Index{2});   \
  }                                                                            \
                                                                               \
  auto label_connected_components(const nd_array<Index> &connectivity,         \
                                  Index expected_number_of_components)         \
      -> connected_components_result<Index> {                                  \
    return detail::label_dense_connected_components(                           \
        connectivity, expected_number_of_components);                          \
  }

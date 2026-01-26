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
#include "./path_connector.hpp"

namespace tf {

/// @ingroup topology_paths
/// @brief Connect edges into continuous vertex paths.
///
/// Takes a collection of edges and connects them into continuous paths.
/// Each path is a sequence of vertices where consecutive vertices are
/// connected by an edge. Handles both open paths and closed loops.
///
/// @tparam Policy The edges policy type.
/// @param edges The edges to connect.
/// @return An @ref tf::offset_block_buffer where each block is a path of vertex indices.
template <typename Policy>
auto connect_edges_to_paths(const tf::edges<Policy> &edges) {
  using Index = std::decay_t<decltype(edges[0][0])>;
  tf::path_connector<Index, Index> pc;
  pc.build(edges);
  return pc.paths_buffer();
}
} // namespace tf

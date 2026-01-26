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
#include "../core/edges.hpp"
#include "../core/faces.hpp"
#include "../core/offset_block_buffer.hpp"
#include "./boundary_edges.hpp"
#include "./face_membership_like.hpp"
#include "./find_eulerian_paths.hpp"
#include "./vertex_link.hpp"

namespace tf {

/// @ingroup topology_analysis
/// @brief Extract boundary paths from edges.
///
/// Finds all boundary edges and connects them into continuous paths
/// using Eulerian path finding. Each path is a sequence of vertex indices
/// forming a connected boundary loop or open path.
///
/// @tparam Policy The edges policy type.
/// @param edges The boundary edges.
/// @param max_vertex_id The maximum vertex id (for internal structures).
/// @return An @ref tf::offset_block_buffer where each block is a boundary path.
template <typename Policy>
auto make_boundary_paths(const tf::edges<Policy> &edges,
                         std::size_t max_vertex_id) {
  using Index = std::decay_t<decltype(edges[0][0])>;
  tf::vertex_link<Index> vl;
  vl.build(edges, max_vertex_id, tf::edge_orientation::forward);
  tf::offset_block_buffer<Index, Index> buffer;
  buffer.data_buffer().reserve(max_vertex_id);
  buffer.offsets_buffer().reserve(3);
  tf::find_eulerian_paths(vl, buffer.offsets_buffer(), buffer.data_buffer());
  return buffer;
}

/// @ingroup topology_analysis
/// @brief Extract boundary paths from faces and face membership.
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @return An @ref tf::offset_block_buffer where each block is a boundary path.
template <typename Policy, typename Policy1>
auto make_boundary_paths(const tf::faces<Policy> &faces,
                         const tf::face_membership_like<Policy1> &fm) {
  auto edges = tf::make_boundary_edges(faces, fm);
  return tf::make_boundary_paths(tf::make_edges(edges), fm.size());
}

/// @ingroup topology_analysis
/// @brief Extract boundary paths from a polygons range.
///
/// Convenience overload that extracts boundary edges and connects them
/// into paths.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return An @ref tf::offset_block_buffer where each block is a boundary path.
template <typename Policy>
auto make_boundary_paths(const tf::polygons<Policy> &polygons) {
  auto edges = tf::make_boundary_edges(polygons);
  return tf::make_boundary_paths(tf::make_edges(edges),
                                 polygons.points().size());
}
} // namespace tf

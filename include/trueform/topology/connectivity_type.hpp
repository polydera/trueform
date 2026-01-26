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

namespace tf {

/// @ingroup topology_types
/// @brief Connectivity types for mesh traversal and component labeling.
///
/// Determines how mesh elements are considered connected:
/// - manifold_edge: connected only through manifold edges (separates at
///   boundaries and non-manifold edges)
/// - edge: connected through any shared edge (ignores non-manifold status)
/// - vertex: connected through any shared vertex (most permissive)
///
/// @see tf::label_connected_components()
/// @see tf::make_vertex_connected_component_labels()
/// @see tf::make_edge_connected_component_labels()
/// @see tf::make_manifold_edge_connected_component_labels()
enum class connectivity_type {
  manifold_edge,  ///< Connect only through manifold (2-face) edges.
  edge,           ///< Connect through any shared edge.
  vertex          ///< Connect through any shared vertex.
};

} // namespace tf

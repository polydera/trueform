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
/// - manifold_edge: faces connect only through manifold (2-face) edges,
///   so boundaries and non-manifold edges separate
/// - edge: faces connect through any shared edge, non-manifold included
/// - vertex: vertices connect through any shared edge; the labels live on
///   vertices, not faces
///
/// @see tf::label_connected_components()
/// @see tf::make_vertex_connected_component_labels()
/// @see tf::make_edge_connected_component_labels()
/// @see tf::make_manifold_edge_connected_component_labels()
enum class connectivity_type {
  manifold_edge,  ///< Connect only through manifold (2-face) edges.
  edge,           ///< Connect through any shared edge.
  vertex          ///< Label vertices, joined through any shared edge.
};

} // namespace tf

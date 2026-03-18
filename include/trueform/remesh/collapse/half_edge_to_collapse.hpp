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

#include "../../topology/half_edges.hpp"
#include "../feature_mask.hpp"

namespace tf::remesh {

/// @ingroup remesh
/// @brief Pick the collapse direction for an edge.
///
/// Returns the half-edge whose start vertex will be removed and whose
/// end vertex will survive. Prefers keeping the boundary vertex.
template <typename Index>
auto half_edge_to_collapse(const tf::half_edges<Index> &he,
                           tf::edge_handle<Index> eh)
    -> tf::half_edge_handle<Index> {
  auto heh0 = he.half_edge_handle(tf::unsafe, eh, false);
  auto v0 = he.start_vertex_handle(tf::unsafe, heh0).id();
  auto v1 = he.end_vertex_handle(tf::unsafe, heh0).id();
  if (he.is_boundary_vertex(v1) && !he.is_boundary_vertex(v0))
    return he.opposite(tf::unsafe, heh0);
  return heh0;
}

/// @ingroup remesh
/// @brief Pick the collapse direction for an edge with feature awareness.
///
/// Returns the half-edge whose start vertex will be removed and whose
/// end vertex will survive. Prefers keeping the more constrained vertex:
/// boundary > corner > crease > regular.
template <typename Index>
auto half_edge_to_collapse(const tf::half_edges<Index> &he,
                           tf::edge_handle<Index> eh,
                           const feature_mask &mask)
    -> tf::half_edge_handle<Index> {
  auto heh0 = he.half_edge_handle(tf::unsafe, eh, false);
  auto v0 = he.start_vertex_handle(tf::unsafe, heh0).id();
  auto v1 = he.end_vertex_handle(tf::unsafe, heh0).id();
  if (he.is_boundary_vertex(v1) && !he.is_boundary_vertex(v0))
    return he.opposite(tf::unsafe, heh0);
  if (he.is_boundary_vertex(v0) && !he.is_boundary_vertex(v1))
    return heh0;
  if (mask.vertex_type(v1) > mask.vertex_type(v0))
    return he.opposite(tf::unsafe, heh0);
  return heh0;
}

} // namespace tf::remesh

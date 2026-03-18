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

#include "../core/algorithm/parallel_copy.hpp"
#include "../core/buffer.hpp"
#include "../core/index_map.hpp"
#include "../core/views/indirect_range.hpp"

#include <cstdint>

namespace tf::remesh {

/// @ingroup remesh
/// @brief Vertex classification relative to feature edges.
enum class vertex_feature_type : std::uint8_t {
  regular = 0, ///< 0 adjacent feature edges — free to move
  crease = 1,  ///< exactly 2 adjacent feature edges — slide along crease
  corner = 2,  ///< 3+ adjacent feature edges — frozen
};

/// @ingroup remesh
/// @brief Per-edge feature flags and per-vertex classification.
///
/// Computed from dihedral angles via @ref tf::make_feature_mask, or
/// constructed manually to mark arbitrary edges as features. Reused
/// by all remesh stages (collapse, valence optimization, relaxation).
///
/// Users can modify @ref edges directly and call
/// @ref tf::remesh::recompute_vertex_types to update vertex
/// classification after manual edits.
struct feature_mask {
  tf::buffer<bool> edges; ///< Per-edge: is this a feature edge?
  tf::buffer<vertex_feature_type> vertices; ///< Per-vertex classification

  auto is_feature(std::size_t edge_id) const -> bool {
    return edges[edge_id];
  }

  auto vertex_type(std::size_t vertex_id) const -> vertex_feature_type {
    return vertices[vertex_id];
  }

  auto is_crease(std::size_t v) const -> bool {
    return vertices[v] == vertex_feature_type::crease;
  }

  auto is_corner(std::size_t v) const -> bool {
    return vertices[v] == vertex_feature_type::corner;
  }

  /// @brief Check if removing a vertex via edge collapse is forbidden.
  ///
  /// A corner vertex must never be removed. A crease vertex can only
  /// be removed along its crease (the edge must be a feature edge).
  /// Regular vertices are never protected.
  ///
  /// @param edge_id The edge being collapsed.
  /// @param v_removed The vertex that will be removed.
  auto is_collapse_forbidden(std::size_t edge_id,
                             std::size_t v_removed) const -> bool {
    auto t = vertices[v_removed];
    if (t == vertex_feature_type::corner)
      return true;
    if (t == vertex_feature_type::crease)
      return !edges[edge_id];
    return false;
  }

  auto empty() const -> bool { return edges.size() == 0; }

  /// @brief Remap the mask to compacted IDs.
  ///
  /// Gathers kept edges and vertices into new compact buffers using
  /// the index maps returned by half_edges::compact().
  template <typename Range0, typename Range1, typename Range2, typename Range3>
  auto compact(const tf::index_map<Range0, Range1> &edge_im,
               const tf::index_map<Range2, Range3> &vert_im) -> void {
    tf::buffer<bool> new_edges;
    new_edges.allocate(edge_im.kept_ids().size());
    tf::parallel_copy(
        tf::make_indirect_range(edge_im.kept_ids(), tf::make_range(edges)),
        tf::make_range(new_edges));
    edges = std::move(new_edges);

    tf::buffer<vertex_feature_type> new_verts;
    new_verts.allocate(vert_im.kept_ids().size());
    tf::parallel_copy(
        tf::make_indirect_range(vert_im.kept_ids(), tf::make_range(vertices)),
        tf::make_range(new_verts));
    vertices = std::move(new_verts);
  }
};

} // namespace tf::remesh

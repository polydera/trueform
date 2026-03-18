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

#include "../core/algorithm/parallel_for_each.hpp"
#include "../core/angle.hpp"
#include "../core/coordinate_type.hpp"
#include "../core/cross.hpp"
#include "../core/dot.hpp"
#include "../core/points.hpp"
#include "../core/vector.hpp"
#include "../core/views/sequence_range.hpp"
#include "../topology/half_edges.hpp"
#include "./feature_mask.hpp"

#include <cmath>

namespace tf::remesh {

/// @ingroup remesh
/// @brief Recompute per-vertex feature classification from edge flags.
///
/// Walks each vertex's 1-ring, counts adjacent feature edges, and
/// classifies: regular (0), crease (2), corner (3+). Vertices with
/// exactly 1 adjacent feature edge ("darts") are classified as regular.
///
/// Call this after manually editing @ref feature_mask::edges to keep
/// vertex classification consistent.
template <typename Index>
auto recompute_vertex_types(const tf::half_edges<Index> &he, feature_mask &mask)
    -> void {
  Index n_verts = Index(he.vertex_half_edge_handles().size());
  if (mask.vertices.size() < std::size_t(n_verts))
    mask.vertices.allocate(n_verts);

  tf::parallel_for_each(tf::make_sequence_range(n_verts), [&](Index v) {
    auto vhe = he.vertex_half_edge_handles()[v];
    if (!vhe.is_valid()) {
      mask.vertices[v] = vertex_feature_type::regular;
      return;
    }
    int count = 0;
    auto cur = vhe;
    do {
      auto eid = he.edge_handle(tf::unsafe, cur).id();
      if (mask.edges[eid])
        ++count;
      cur = he.rotated(cur);
      if (!cur.is_valid())
        break;
    } while (cur != vhe);

    if (count >= 3)
      mask.vertices[v] = vertex_feature_type::corner;
    else if (count == 2)
      mask.vertices[v] = vertex_feature_type::crease;
    else
      mask.vertices[v] = vertex_feature_type::regular;
  });
}

} // namespace tf::remesh

namespace tf {

/// @ingroup remesh
/// @brief Compute a feature mask from a dihedral angle threshold.
///
/// An interior edge is marked as a feature when the dihedral angle
/// between its two adjacent face normals is sharper than
/// @p feature_angle. Boundary edges are never features.
///
/// Vertex classification is derived by counting adjacent feature edges:
/// regular (0), crease (2), corner (3+).
///
/// @param he The half-edge structure.
/// @param points Vertex positions.
/// @param feature_angle Threshold angle. Edges with dihedral angle
///        below this are marked as features.
/// @return A @ref tf::remesh::feature_mask with per-edge flags and
///         per-vertex classification.
template <typename Index, typename PointsPolicy>
auto make_feature_mask(const tf::half_edges<Index> &he,
                       const tf::points<PointsPolicy> &points,
                       tf::rad<tf::coordinate_type<PointsPolicy>> feature_angle)
    -> tf::remesh::feature_mask {
  Index n_edges = Index(he.half_edges_buffer().size() / 2);

  double cos_threshold = std::cos(double(feature_angle.value));

  tf::remesh::feature_mask mask;
  mask.edges.allocate(n_edges);

  tf::parallel_for_each(tf::make_sequence_range(n_edges), [&](Index eid) {
    auto eh = tf::edge_handle<Index>{eid};
    auto heh0 = he.half_edge_handle(tf::unsafe, eh, false);
    auto heh1 = he.opposite(tf::unsafe, heh0);

    if (!heh0.is_valid() || !he.is_simple(tf::unsafe, heh0) ||
        !heh1.is_valid() || !he.is_simple(tf::unsafe, heh1)) {
      mask.edges[eid] = false;
      return;
    }

    // Compute face normal 0
    auto h0a = heh0;
    auto h0b = he.next(tf::unsafe, h0a);
    auto h0c = he.next(tf::unsafe, h0b);
    auto p0a = points[he.start_vertex_handle(tf::unsafe, h0a).id()];
    auto p0b = points[he.start_vertex_handle(tf::unsafe, h0b).id()];
    auto p0c = points[he.start_vertex_handle(tf::unsafe, h0c).id()];
    auto e0a = tf::make_vector(double(p0b[0] - p0a[0]), double(p0b[1] - p0a[1]),
                               double(p0b[2] - p0a[2]));
    auto e0b = tf::make_vector(double(p0c[0] - p0a[0]), double(p0c[1] - p0a[1]),
                               double(p0c[2] - p0a[2]));
    auto n0 = tf::cross(e0a, e0b);

    // Compute face normal 1
    auto h1a = heh1;
    auto h1b = he.next(tf::unsafe, h1a);
    auto h1c = he.next(tf::unsafe, h1b);
    auto p1a = points[he.start_vertex_handle(tf::unsafe, h1a).id()];
    auto p1b = points[he.start_vertex_handle(tf::unsafe, h1b).id()];
    auto p1c = points[he.start_vertex_handle(tf::unsafe, h1c).id()];
    auto e1a = tf::make_vector(double(p1b[0] - p1a[0]), double(p1b[1] - p1a[1]),
                               double(p1b[2] - p1a[2]));
    auto e1b = tf::make_vector(double(p1c[0] - p1a[0]), double(p1c[1] - p1a[1]),
                               double(p1c[2] - p1a[2]));
    auto n1 = tf::cross(e1a, e1b);

    // Normalize and compare
    double len0 = n0.length();
    double len1 = n1.length();
    if (len0 < 1e-30 || len1 < 1e-30) {
      mask.edges[eid] = false;
      return;
    }

    double cos_dihedral = tf::dot(n0, n1) / (len0 * len1);
    mask.edges[eid] = cos_dihedral < cos_threshold;
  });

  tf::remesh::recompute_vertex_types(he, mask);
  return mask;
}

/// @ingroup remesh
/// @brief Compute a feature mask from a dihedral angle threshold in degrees.
/// @overload
template <typename Index, typename PointsPolicy>
auto make_feature_mask(const tf::half_edges<Index> &he,
                       const tf::points<PointsPolicy> &points,
                       tf::deg<tf::coordinate_type<PointsPolicy>> feature_angle)
    -> tf::remesh::feature_mask {
  using T = tf::coordinate_type<PointsPolicy>;
  return tf::make_feature_mask(he, points, tf::rad<T>{feature_angle});
}

} // namespace tf

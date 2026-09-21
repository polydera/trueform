/*
 * Copyright (c) 2026 XLAB
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
#include "../core/blocked_buffer.hpp"
#include "../core/buffer.hpp"
#include "../core/edges.hpp"
#include "../core/faces.hpp"
#include "../core/offset_block_buffer.hpp"
#include "../core/polygons.hpp"
#include "../core/views/zip.hpp"
#include "./boundary/gather_boundary_edges.hpp"
#include "./edge_path_connector.hpp"
#include "./face_membership.hpp"
#include "./face_membership_like.hpp"
#include "./policy/face_membership.hpp"
#include <cstddef>
#include <type_traits>

namespace tf {

/// @ingroup topology_analysis
/// @brief A mesh's boundary rims: the vertices of each, the face across
///        each of its edges, and whether it closes.
///
/// Block `i` of `vertices` is rim `i` walked, and block `i` of `faces`
/// names the face carrying each of its edges, so rim edge `k` runs from
/// vertex `k` to vertex `k + 1` and is carried by face `k` alone. A closed
/// rim of `n` vertices has `n` edges, the last running back to vertex `0`;
/// an open one has `n - 1`.
///
/// @tparam Index The integer type for vertex and face indices.
template <typename Index> struct boundary_rims {
  /// @brief Rim `i`'s vertex ids, in the order it is walked.
  tf::offset_block_buffer<Index, Index> vertices;
  /// @brief The face carrying each of rim `i`'s edges.
  tf::offset_block_buffer<Index, Index> faces;
  /// @brief Whether rim `i`'s last edge runs back to its first vertex.
  tf::buffer<bool> closed;

  /// @brief The number of rims.
  auto size() const -> std::size_t { return closed.size(); }
};

/// @ingroup topology_analysis
/// @brief Assemble a mesh's boundary edges into rims.
///
/// A rim ends where the boundary stops passing straight through, so a
/// vertex it reaches more than twice closes one rim and starts the next
/// and a pinched boundary comes back as its pieces rather than as one
/// figure eight. The mesh decides the rest: the boundary edges are its own
/// order and the walk takes them in it, reaching a rim with two ends before
/// the closed ones. A rim keeps the walk's direction when the edge the walk
/// entered it by runs the way its face wound it and takes the opposite
/// otherwise, so a consistently wound mesh gives a rim that runs the way
/// its faces wind their boundary throughout.
///
/// @tparam Policy The faces policy type.
/// @tparam Policy1 The face membership policy type.
/// @param faces The faces range.
/// @param fm The face membership structure.
/// @return The @ref tf::boundary_rims of the mesh.
template <typename Policy, typename Policy1>
auto make_boundary_rims(const tf::faces<Policy> &faces,
                        const tf::face_membership_like<Policy1> &fm) {
  using Index = std::decay_t<decltype(fm[0][0])>;
  tf::blocked_buffer<Index, 2> edges;
  tf::buffer<Index> edge_faces;
  tf::topology::gather_boundary_edges(faces, fm, edges.data_buffer(),
                                      edge_faces);

  tf::edge_path_connector<Index, Index> connector;
  connector.build(tf::make_edges(edges));

  tf::boundary_rims<Index> rims;
  auto &vertex_data = rims.vertices.data_buffer();
  auto &face_data = rims.faces.data_buffer();
  vertex_data.reserve(edges.size() + connector.paths().size());
  face_data.reserve(edges.size());

  for (auto rim : tf::zip(connector.paths(), connector.directions())) {
    auto &&[path, direction] = rim;
    const Index n = Index(path.size());
    const bool forward = direction[0];

    rims.vertices.offsets_buffer().push_back(Index(vertex_data.size()));
    rims.faces.offsets_buffer().push_back(Index(face_data.size()));
    const auto first = vertex_data.size();

    for (Index k = 0; k < n; ++k) {
      const Index position = forward ? k : n - 1 - k;
      const Index e = Index(path[position]);
      const bool as_wound = direction[position] == forward;
      const auto edge = edges[std::size_t(e)];
      if (k == 0)
        vertex_data.push_back(as_wound ? edge[0] : edge[1]);
      vertex_data.push_back(as_wound ? edge[1] : edge[0]);
      face_data.push_back(edge_faces[std::size_t(e)]);
    }

    const bool closed = vertex_data.back() == vertex_data[first];
    if (closed)
      vertex_data.erase_till_end(vertex_data.end() - 1);
    rims.closed.push_back(closed);
  }

  if (rims.closed.size()) {
    rims.vertices.offsets_buffer().push_back(Index(vertex_data.size()));
    rims.faces.offsets_buffer().push_back(Index(face_data.size()));
  }
  return rims;
}

/// @ingroup topology_analysis
/// @brief Assemble a polygons range's boundary edges into rims.
///
/// Convenience overload that builds face membership internally if not
/// provided via policy.
///
/// @tparam Policy The polygons policy type.
/// @param polygons The polygons range.
/// @return The @ref tf::boundary_rims of the mesh.
template <typename Policy>
auto make_boundary_rims(const tf::polygons<Policy> &polygons) {
  if constexpr (tf::has_face_membership_policy<Policy>) {
    return tf::make_boundary_rims(polygons.faces(), polygons.face_membership());
  } else {
    tf::face_membership<std::decay_t<decltype(polygons.faces()[0][0])>> fm;
    fm.build(polygons);
    return tf::make_boundary_rims(polygons.faces(), fm);
  }
}

} // namespace tf

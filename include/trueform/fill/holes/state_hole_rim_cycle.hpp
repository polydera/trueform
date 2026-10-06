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
#include "../../core/polygons.hpp"
#include "../../topology/boundary_rims.hpp"
#include "../hole_split.hpp"
#include "./hole_carrier_normal.hpp"
#include "./hole_lattice.hpp"
#include "./hole_rim.hpp"
#include "./validate_hole_rim.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace tf::fill {

/// State a validated rim's cycle, ready for filling.
///
/// The cycle is rooted at the rim's smallest vertex and turned the way the
/// elected ticket states: the ticket of the smaller-keyed of the two rim
/// edges at that root, whose carrying face's own winding decides the
/// direction the whole cycle takes.
template <typename Policy, typename Index, typename Int, typename RealT>
auto state_hole_rim_cycle(const tf::polygons<Policy> &polygons,
                          const tf::boundary_rims<Index> &rims, Index rim,
                          const tf::fill::hole_rim_scratch<Index, Int> &scratch,
                          tf::fill::hole_rim<Index, Int, RealT> &out) -> void {
  const auto vertices = rims.vertices[std::size_t(rim)];
  const auto carriers = rims.faces[std::size_t(rim)];
  const auto mesh_faces = polygons.faces();
  const auto points = polygons.points();
  const Index n = Index(vertices.size());
  const auto next = [n](Index k) { return Index(k + 1 == n ? 0 : k + 1); };

  Index root = 0;
  for (Index k = 1; k < n; ++k)
    if (vertices[std::size_t(k)] < vertices[std::size_t(root)])
      root = k;
  const auto key = [&vertices, next](Index e) {
    const Index a = vertices[std::size_t(e)];
    const Index b = vertices[std::size_t(next(e))];
    return std::pair<Index, Index>{std::min(a, b), std::max(a, b)};
  };
  const Index before = Index(root == 0 ? n - 1 : root - 1);
  const Index elected = key(before) < key(root) ? before : root;
  const bool turn = !scratch.forward[std::size_t(elected)];

  out.clear();
  out.sites.allocate(std::size_t(n));
  out.positions.allocate(std::size_t(n));
  out.corners.allocate(std::size_t(n));
  out.edges.allocate(std::size_t(n));
  for (Index j = 0; j < n; ++j) {
    const Index k =
        turn ? Index((root + n - j) % n) : Index((root + j) % n);
    const Index vertex = vertices[std::size_t(k)];
    out.sites[std::size_t(j)] = {vertex, tf::fill::hole_scaled_point<Int>(
                                             scratch.lattice[std::size_t(k)])};
    out.positions[std::size_t(j)] = points[std::size_t(vertex)];
    out.corners[std::size_t(j)] = vertex;

    const Index e = turn ? Index((k + n - 1) % n) : k;
    const Index a = vertices[std::size_t(e)];
    const Index b = vertices[std::size_t(next(e))];
    auto normal = tf::fill::hole_carrier_normal(mesh_faces, points,
                                                carriers[std::size_t(e)]);
    const bool forward = turn ? !scratch.forward[std::size_t(e)]
                              : scratch.forward[std::size_t(e)];
    if (!forward)
      normal = -normal;
    const Index v0 = std::min(a, b), v1 = std::max(a, b);
    const std::uint8_t start =
        std::uint8_t((turn ? b : a) == v0 ? 0 : tf::hole_split_scale);
    out.edges[std::size_t(j)] = {carriers[std::size_t(e)],
                                 v0,
                                 v1,
                                 normal,
                                 start,
                                 std::uint8_t(tf::hole_split_scale - start),
                                 forward};
  }
}

} // namespace tf::fill

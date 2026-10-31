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
#include "../../core/buffer.hpp"
#include "../../core/polygons.hpp"
#include "../../topology/face_membership_like.hpp"
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>

namespace tf::fill {

/// The host neighbourhood a chart continues: the rim, then two rings of host
/// faces outward from it through face membership — a topological collar,
/// never a distance query.
///
/// `vertices` holds the rim in its own order, then the ring-1 vertices, then
/// the ring-2 ones, each ring ascending; ring `r` is
/// `[vertex_offsets[r], vertex_offsets[r + 1])`, the rim being ring 0, and a
/// vertex's position in `vertices` is its COLLAR POSITION. `faces` holds the
/// ring-1 faces — every face at a rim vertex — then the ring-2 ones, each
/// ring ascending and split at `face_offsets[1]`. `ring_triangles` states
/// each ring-1 face's corners as collar positions in the host's winding;
/// those faces are the only ones whose corners the collar holds whole.
///
/// @tparam Index The mesh's index type.
template <typename Index> struct hole_collar {
  tf::buffer<Index> vertices;
  std::array<Index, 4> vertex_offsets;
  tf::buffer<Index> faces;
  std::array<Index, 3> face_offsets;
  tf::buffer<std::array<Index, 3>> ring_triangles;

  auto n_rim() const -> std::size_t { return std::size_t(vertex_offsets[1]); }
};

/// The scratch one collar's gather walks on.
template <typename Index> struct hole_collar_scratch {
  tf::buffer<std::array<Index, 2>> rim_positions;
  tf::buffer<Index> found;
  tf::buffer<std::array<Index, 2>> corners;
};

/// Gather the collar of one rim, the rim given as host vertex ids in its own
/// order, each named once — the fact preflight establishes.
///
/// Every membership question is a batch of queries sorted and merged
/// against the sorted rings it asks about, so the gather costs the collar's
/// size times the logarithm of it in sorts and linear merges, and never
/// touches the mesh beyond the faces it names. Returns false where a collar
/// face is not a triangle.
template <typename Policy, typename MembershipPolicy, typename RimRange,
          typename Index>
auto gather_hole_collar(const tf::polygons<Policy> &polygons,
                        const tf::face_membership_like<MembershipPolicy> &fm,
                        const RimRange &rim,
                        tf::fill::hole_collar_scratch<Index> &scratch,
                        tf::fill::hole_collar<Index> &collar) -> bool {
  const auto mesh_faces = polygons.faces();
  const Index n = Index(rim.size());

  scratch.rim_positions.clear();
  for (Index k = 0; k < n; ++k)
    scratch.rim_positions.push_back({Index(rim[std::size_t(k)]), k});
  tbb::parallel_sort(scratch.rim_positions.begin(), scratch.rim_positions.end());
  const auto ascending = [](tf::buffer<Index> &ids, std::size_t from) {
    tbb::parallel_sort(ids.begin() + from, ids.end());
    ids.erase_till_end(std::unique(ids.begin() + from, ids.end()));
  };

  collar.vertices.clear();
  for (Index k = 0; k < n; ++k)
    collar.vertices.push_back(Index(rim[std::size_t(k)]));
  collar.faces.clear();
  for (Index k = 0; k < n; ++k)
    for (auto face : fm[std::size_t(rim[std::size_t(k)])])
      collar.faces.push_back(Index(face));
  ascending(collar.faces, 0);
  const Index ring1_faces = Index(collar.faces.size());

  scratch.corners.clear();
  for (Index f = 0; f < ring1_faces; ++f) {
    const auto corners = mesh_faces[std::size_t(collar.faces[std::size_t(f)])];
    if (corners.size() != 3)
      return false;
    for (std::size_t c = 0; c < 3; ++c)
      scratch.corners.push_back({Index(corners[c]), Index(3 * f + Index(c))});
  }
  tbb::parallel_sort(scratch.corners.begin(), scratch.corners.end());

  collar.ring_triangles.allocate(std::size_t(ring1_faces));
  {
    std::size_t r = 0;
    Index last = Index(-1);
    for (const auto &corner : scratch.corners) {
      while (r < std::size_t(n) && scratch.rim_positions[r][0] < corner[0])
        ++r;
      Index position;
      if (r < std::size_t(n) && scratch.rim_positions[r][0] == corner[0])
        position = scratch.rim_positions[r][1];
      else {
        if (corner[0] != last)
          collar.vertices.push_back(corner[0]);
        last = corner[0];
        position = Index(collar.vertices.size() - 1);
      }
      collar.ring_triangles[std::size_t(corner[1] / 3)]
                           [std::size_t(corner[1] % 3)] = position;
    }
  }
  const Index ring1_end = Index(collar.vertices.size());

  const std::size_t ring2_from = collar.faces.size();
  for (auto at = collar.vertices.begin() + n;
       at != collar.vertices.begin() + ring1_end; ++at)
    for (auto face : fm[std::size_t(*at)])
      collar.faces.push_back(Index(face));
  ascending(collar.faces, ring2_from);
  {
    std::size_t write = ring2_from, i = 0;
    for (std::size_t f = ring2_from; f < collar.faces.size(); ++f) {
      const Index face = collar.faces[f];
      while (i < std::size_t(ring1_faces) && collar.faces[i] < face)
        ++i;
      if (i < std::size_t(ring1_faces) && collar.faces[i] == face)
        continue;
      collar.faces[write++] = face;
    }
    collar.faces.erase_till_end(collar.faces.begin() + std::ptrdiff_t(write));
  }

  scratch.found.clear();
  for (std::size_t f = std::size_t(ring1_faces); f < collar.faces.size(); ++f) {
    const auto corners = mesh_faces[std::size_t(collar.faces[f])];
    if (corners.size() != 3)
      return false;
    for (std::size_t c = 0; c < 3; ++c)
      scratch.found.push_back(Index(corners[c]));
  }
  ascending(scratch.found, 0);
  {
    std::size_t r = 0, q = std::size_t(n);
    for (auto vertex : scratch.found) {
      while (r < std::size_t(n) && scratch.rim_positions[r][0] < vertex)
        ++r;
      if (r < std::size_t(n) && scratch.rim_positions[r][0] == vertex)
        continue;
      while (q < std::size_t(ring1_end) && collar.vertices[q] < vertex)
        ++q;
      if (q < std::size_t(ring1_end) && collar.vertices[q] == vertex)
        continue;
      collar.vertices.push_back(vertex);
    }
  }

  collar.vertex_offsets = {Index(0), n, ring1_end,
                           Index(collar.vertices.size())};
  collar.face_offsets = {Index(0), ring1_faces, Index(collar.faces.size())};
  return true;
}

} // namespace tf::fill

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
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/faces.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/unsafe.hpp"
#include "../../topology/face_membership.hpp"
#include "../../topology/half_edges.hpp"
#include "./hole_metric.hpp"
#include "./hole_rim.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::fill {

/// One coarse patch read as a mesh of its own.
///
/// A patch vertex is its flat identity, numbered by ascending flat id, so two
/// runs of one rim name the same vertex alike; `positions` and `carrier` are
/// that numbering's own lanes — the carrying triangle a rim split inherits,
/// `-1` for an original vertex and for a point minted inside the patch.
///
/// The half-edge structure is the topology's authority once it is built, and
/// its face and vertex buffer lengths are the counts a refinement reads: the
/// moves append and never remove, and `rebuild_handles` restates the cached
/// counts and the representatives when the refinement ends.
///
/// @tparam Index The mesh's index type.
/// @tparam RealT The mesh's coordinate type.
template <typename Index, typename RealT> struct hole_patch {
  tf::points_buffer<RealT, 3> positions;
  tf::buffer<Index> flat;
  tf::buffer<Index> carrier;
  tf::half_edges<Index> he;

  auto n_vertices() const -> std::size_t { return flat.size(); }

  auto n_faces() const -> std::size_t {
    return he.face_half_edge_handles_buffer().size();
  }

  /// The face's three corners, in its own winding.
  auto corners_of(Index face) const -> std::array<Index, 3> {
    const auto h0 = he.face_half_edge_handles()[std::size_t(face)];
    const auto h1 = he.next(tf::unsafe, h0);
    const auto h2 = he.next(tf::unsafe, h1);
    return {he.start_vertex_handle(tf::unsafe, h0).id(),
            he.start_vertex_handle(tf::unsafe, h1).id(),
            he.start_vertex_handle(tf::unsafe, h2).id()};
  }
};

/// The scratch one patch's build walks on, reused across the rims a worker
/// takes.
template <typename Index> struct hole_patch_scratch {
  tf::buffer<Index> names;
  tf::buffer<std::array<Index, 3>> normalized;
  tf::blocked_buffer<Index, 3> faces;
  tf::face_membership<Index> membership;
};

/// The patch of a prepared rim and the triangles a tier stated over it.
///
/// The triangles arrive in the tier's own order and leave in the canonical
/// one — each rotated onto its smallest corner with its winding kept, the
/// block then sorted — so the structure a patch carries is the rim's and
/// never the tier's enumeration.
template <typename Index, typename Int, typename RealT>
auto build_hole_patch(const tf::fill::hole_rim<Index, Int, RealT> &rim,
                      const tf::buffer<std::array<Index, 3>> &triangles,
                      tf::fill::hole_patch_scratch<Index> &scratch,
                      tf::fill::hole_patch<Index, RealT> &patch) -> void {
  scratch.names.clear();
  for (const auto &triangle : triangles)
    for (int corner = 0; corner < 3; ++corner)
      scratch.names.push_back(triangle[std::size_t(corner)]);
  std::sort(scratch.names.begin(), scratch.names.end());
  scratch.names.erase_till_end(
      std::unique(scratch.names.begin(), scratch.names.end()));

  const auto local_of = [&scratch](Index name) {
    return Index(std::lower_bound(scratch.names.begin(), scratch.names.end(),
                                  name) -
                 scratch.names.begin());
  };
  const std::size_t n = scratch.names.size();

  patch.flat.clear();
  for (auto name : scratch.names)
    patch.flat.push_back(name);
  patch.positions.allocate(n);
  patch.carrier.allocate(n);
  for (std::size_t k = 0; k < n; ++k)
    patch.carrier[k] = Index(-1);
  for (Index k = 0; k < Index(rim.size()); ++k) {
    const std::size_t local =
        std::size_t(local_of(rim.corners[std::size_t(k)]));
    patch.positions[local] = rim.positions[std::size_t(k)];
    if (tf::fill::hole_position_is_split(rim, k))
      patch.carrier[local] = rim.edges[std::size_t(k)].face;
  }

  scratch.normalized.clear();
  for (const auto &triangle : triangles) {
    const std::array<Index, 3> local{local_of(triangle[0]),
                                     local_of(triangle[1]),
                                     local_of(triangle[2])};
    const int first = tf::fill::hole_canonical_rotation(local[0], local[1],
                                                        local[2]);
    scratch.normalized.push_back({local[std::size_t(first)],
                                  local[std::size_t((first + 1) % 3)],
                                  local[std::size_t((first + 2) % 3)]});
  }
  std::sort(scratch.normalized.begin(), scratch.normalized.end());

  scratch.faces.clear();
  for (const auto &triangle : scratch.normalized)
    scratch.faces.push_back(triangle);
  const auto faces = tf::make_faces(scratch.faces);
  scratch.membership.build(faces, n);
  patch.he.build(faces, scratch.membership);
}

/// State a patch back in flat identities, in the canonical order.
template <typename Index, typename RealT>
auto emit_hole_patch(const tf::fill::hole_patch<Index, RealT> &patch,
                     tf::fill::hole_patch_scratch<Index> &scratch,
                     tf::buffer<std::array<Index, 3>> &out) -> void {
  scratch.normalized.clear();
  for (std::size_t face = 0; face < patch.n_faces(); ++face) {
    const auto local = patch.corners_of(Index(face));
    const std::array<Index, 3> named{patch.flat[std::size_t(local[0])],
                                     patch.flat[std::size_t(local[1])],
                                     patch.flat[std::size_t(local[2])]};
    const int first =
        tf::fill::hole_canonical_rotation(named[0], named[1], named[2]);
    scratch.normalized.push_back({named[std::size_t(first)],
                                  named[std::size_t((first + 1) % 3)],
                                  named[std::size_t((first + 2) % 3)]});
  }
  std::sort(scratch.normalized.begin(), scratch.normalized.end());
  out.clear();
  for (const auto &triangle : scratch.normalized)
    out.push_back(triangle);
}

} // namespace tf::fill

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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::arrangement {

/// The definitions a MESH states: one boundary side per corner, read straight
/// off the polygon. Nothing has been retired and no root is cut, so a side
/// covers its original edge whole and its endpoints are the two corner ids in
/// key order — with one form there is one tag, so key order IS ascending id.
///
/// A face emits exactly as many rows as it has corners, so counts and one
/// prefix hand every face a disjoint slice. The slice must be known before the
/// row is written: a definition carries its own emission row as `id`, which is
/// what @ref tf::intersect::graph::canonicalize_plane_edge_defs rewrites into
/// the canonical group and what the plane CSR resolves through.
template <typename Index, typename Faces>
auto build_mesh_plane_defs(
    const Faces &faces,
    tf::buffer<tf::intersect::graph::plane_edge_def<Index>> &defs,
    tf::buffer<Index> &face_def_offsets) -> void {
  const auto n_faces = Index(faces.size());
  face_def_offsets.allocate(std::size_t(n_faces) + 1);
  face_def_offsets[0] = Index(0);
  tf::parallel_for_each(
      tf::make_sequence_range(n_faces),
      [&](Index face) {
        face_def_offsets[std::size_t(face) + 1] =
            Index(faces[std::size_t(face)].size());
      },
      tf::checked);
  for (std::size_t face = 1; face <= std::size_t(n_faces); ++face)
    face_def_offsets[face] += face_def_offsets[face - 1];

  defs.allocate(std::size_t(face_def_offsets[std::size_t(n_faces)]));
  tf::parallel_for_each(
      tf::make_sequence_range(n_faces),
      [&](Index face) {
        const auto corners = faces[std::size_t(face)];
        const auto base = std::size_t(face_def_offsets[std::size_t(face)]);
        for (std::size_t side = 0; side < corners.size(); ++side)
          defs[base + side] =
              tf::intersect::graph::make_plane_boundary_side_def<Index>(
                  std::int16_t(0), corners, side, Index(base + side), face,
                  face);
      },
      tf::checked);
}

} // namespace tf::arrangement

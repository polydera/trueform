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

#include "../../arrangement/planes/plane_arrangement.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include <cstddef>

namespace tf::csg::graph {

/// The face each triangle came out of — the inverse of `face_range`.
///
/// The emitting face owns a contiguous span and the spans of one plane's
/// members are disjoint, so the inverse is one scatter over the faces. A row
/// no member's span covers is not written; emission covers every row it
/// appends.
template <typename Index, typename Int>
auto make_plane_triangle_faces(
    const tf::arrangement::plane_arrangement<Index, Int> &arrangement)
    -> tf::buffer<Index> {
  tf::buffer<Index> face_of;
  face_of.allocate(arrangement.triangles().size());
  tf::parallel_for_each(
      tf::make_sequence_range(arrangement.n_faces()),
      [&](Index face) {
        const auto range = arrangement.face_range(face);
        for (auto t = range[0]; t < range[1]; ++t)
          face_of[std::size_t(t)] = face;
      },
      tf::checked);
  return face_of;
}

} // namespace tf::csg::graph

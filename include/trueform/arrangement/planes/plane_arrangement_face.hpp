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

#include "../../exact/plane_frame.hpp"
#include "../../intersect/graph/face_descriptor.hpp"
#include "./plane_arrangement.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::arrangement {

/// What states an arrangement face: below the arrangement's base extent the
/// world does, at or past it the arrangement's own promoted row — the halves
/// the promotion appended.
template <typename Index, typename Int, typename World>
auto plane_arrangement_face_descriptor(
    const plane_arrangement<Index, Int> &arrangement, const World &world,
    Index face) -> const tf::intersect::graph::face_descriptor<Index> & {
  const auto base = arrangement.n_base_faces();
  return face < base
             ? world.descriptor(face)
             : arrangement.promoted_descriptors()[std::size_t(face - base)];
}

/// The plane the face's triangles live in, across the same seam: the world
/// states its own, and a promoted face's plane follows its row.
template <typename Index, typename Int, typename World>
auto plane_arrangement_face_plane(
    const plane_arrangement<Index, Int> &arrangement, const World &world,
    Index face) -> Index {
  const auto base = arrangement.n_base_faces();
  return face < base ? world.plane_of_face(face)
                     : arrangement.n_base_planes() + (face - base);
}

/// The carrier's exact frame across the same seam. `plane_n` is the
/// carrier's own normal, and `tf::exact::projection_axes` picks the frame's
/// axes so a loop winds positively in them exactly when its normal points
/// along it — which is what makes the stored winding a SIDE of this normal.
template <typename Index, typename Int, typename World>
auto plane_arrangement_plane_frame(
    const plane_arrangement<Index, Int> &arrangement, const World &world,
    Index plane) -> const tf::exact::plane_frame<Int> & {
  const auto base = arrangement.n_base_planes();
  return plane < base
             ? world.frame(plane)
             : arrangement.promoted_frames()[std::size_t(plane - base)];
}

/// A member's winding in its carrier's projection, across the same seam:
/// `+1` when the face's normal points along the carrier's, `-1` when it
/// opposes it.
template <typename Index, typename Int, typename World>
auto plane_arrangement_face_orientation(
    const plane_arrangement<Index, Int> &arrangement, const World &world,
    Index face) -> std::int8_t {
  const auto base = arrangement.n_base_faces();
  return face < base
             ? world.face_orientation(face)
             : arrangement.promoted_orientations()[std::size_t(face - base)];
}

} // namespace tf::arrangement

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

#include "../../exact/canonical_plane.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/orient3d.hpp"
#include "../../exact/plane_support.hpp"
#include "../../exact/vertex.hpp"

#include <array>
#include <cstddef>

namespace tf::exact {

/// Plane info for a convex face: 2D projection axes + indices of 3
/// non-collinear vertices that define the supporting plane.
struct face_plane_info {
  int ax0, ax1;
  std::size_t i0, i1, i2;
  bool valid;
};

/// face_plane_info plus the signed supporting-plane normal (an orient3d_plane),
/// both read off the single cross product @ref tf::exact::plane_support
/// computes while it searches — so the reject's sign mask reuses it instead of
/// recomputing the cross.
template <typename Int> struct face_plane {
  face_plane_info info;
  orient3d_plane<Int> plane;
};

/// THE CARRIER'S NAME — the ONE predicate that decides whether two faces
/// are COPLANAR, and the same predicate, through the same producer, that
/// the plane graph pools carriers by, so a name stated here is identical
/// by construction to the name stated there for the same face.
///
/// It is ASKED FOR, never carried: naming reduces a wide integer, only a
/// coplanarity decision reads the answer, and the plane record above is
/// touched once per face of every candidate block. So the name is not a
/// field of it — the classifier asks on the route where the question is
/// actually live, and every other face pays nothing.
///
/// The support is restated from the three corners the plane recorded, in
/// the order it accepted them, so the producer reads the carrier's own
/// support and answers exactly what the support scan would have.
template <typename Index, typename Int>
auto face_plane_name(const face_plane<Int> &fp, vertex_range<Index, Int> face)
    -> canonical_plane<Int> {
  plane_support<Int> support;
  support.offer(face[fp.info.i0].pt);
  support.offer(face[fp.info.i1].pt);
  support.offer(face[fp.info.i2].pt);
  return make_canonical_plane<Int>(support);
}

/// The 2D projection axes and the signed supporting-plane normal, on the
/// three vertices the support search kept — the one producer of the
/// complete plane fact.
template <typename Index, typename Int>
auto make_face_plane(vertex_range<Index, Int> face) -> face_plane<Int> {
  using T2 = typename meta<Int>::T2;
  plane_support<Int> support;
  std::array<std::size_t, 3> at{};
  const auto n = face.size();
  for (std::size_t corner = 0; corner < n && support.size < 3; ++corner)
    if (support.offer(face[corner].pt))
      at[std::size_t(support.size - 1)] = corner;
  if (support.size < 3)
    return {{0, 1, 0, 0, 0, false}, {}};
  const T2 nx = support.normal[0], ny = support.normal[1],
           nz = support.normal[2];
  orient3d_plane<Int> plane{face[at[0]].pt, {nx, ny, nz}};
  T2 ax = nx < 0 ? -nx : nx, ay = ny < 0 ? -ny : ny, az = nz < 0 ? -nz : nz;
  face_plane_info info =
      (az >= ax && az >= ay) ? face_plane_info{0, 1, at[0], at[1], at[2], true}
      : (ay >= ax)           ? face_plane_info{0, 2, at[0], at[1], at[2], true}
                             : face_plane_info{1, 2, at[0], at[1], at[2], true};
  return {info, plane};
}

} // namespace tf::exact

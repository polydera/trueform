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

/// What the coplanar kernels read of a plane: its 2D projection axes and
/// whether the face has one. Read off the normal by
/// @ref tf::exact::make_face_plane_info, on the route that projects.
struct face_plane_info {
  int ax0, ax1;
  bool valid;
};

/// A face's supporting plane, stated as its signed normal alone: the cross
/// @ref tf::exact::plane_support computes over the face's corners in face
/// order. The support's first member is the first corner offered, so the
/// plane stands on the face's own first corner and the face holds its base;
/// a face with no plane holds the zero normal. Everything else a kernel reads
/// of the plane is a reading of this normal or of the support, asked for on
/// the route that reads it.
template <typename Int> struct face_plane {
  std::array<typename meta<Int>::T2, 3> normal;

  auto valid() const -> bool {
    using T2 = typename meta<Int>::T2;
    return normal[0] != T2(0) || normal[1] != T2(0) || normal[2] != T2(0);
  }
};

/// The support scan over a face's corners in face order — the one producer
/// of the plane fact — and in `at` the corner each member was offered from.
template <typename Index, typename Int>
auto scan_face_support(vertex_range<Index, Int> face,
                       std::array<std::size_t, 3> &at) -> plane_support<Int> {
  plane_support<Int> support;
  const auto n = face.size();
  for (std::size_t corner = 0; corner < n && support.size < 3; ++corner)
    if (support.offer(face[corner].pt))
      at[std::size_t(support.size - 1)] = corner;
  return support;
}

template <typename Index, typename Int>
auto make_face_plane(vertex_range<Index, Int> face) -> face_plane<Int> {
  std::array<std::size_t, 3> at{};
  return {scan_face_support(face, at).normal};
}

/// The plane as @ref tf::exact::orient3d_plane_sign reads it, standing on
/// the face's first corner. A face with no plane states none, and reads no
/// corner to say so.
template <typename Index, typename Int>
auto face_orient3d_plane(const face_plane<Int> &fp,
                         vertex_range<Index, Int> face) -> orient3d_plane<Int> {
  if (!fp.valid())
    return {};
  return {face[0].pt, fp.normal};
}

/// The projection axes drop the normal's dominant component.
template <typename Int>
auto make_face_plane_info(const face_plane<Int> &fp) -> face_plane_info {
  using T2 = typename meta<Int>::T2;
  if (!fp.valid())
    return {0, 1, false};
  const T2 nx = fp.normal[0], ny = fp.normal[1], nz = fp.normal[2];
  const T2 ax = nx < 0 ? -nx : nx, ay = ny < 0 ? -ny : ny,
           az = nz < 0 ? -nz : nz;
  return (az >= ax && az >= ay) ? face_plane_info{0, 1, true}
         : (ay >= ax)           ? face_plane_info{0, 2, true}
                                : face_plane_info{1, 2, true};
}

/// THE CARRIER'S NAME — the ONE predicate that decides whether two faces
/// are COPLANAR, and the same predicate, through the same producer, that
/// the plane graph pools carriers by, so a name stated here is identical
/// by construction to the name stated there for the same face.
///
/// It is ASKED FOR, never carried: naming reduces a wide integer and only a
/// coplanarity decision reads the answer, so the classifier names the
/// support of @ref tf::exact::scan_face_support on the route where the
/// question is actually live, and every other face pays nothing.
template <typename Int>
auto face_plane_name(const plane_support<Int> &support)
    -> canonical_plane<Int> {
  return make_canonical_plane<Int>(support);
}

} // namespace tf::exact

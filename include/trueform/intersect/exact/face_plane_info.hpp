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

#include "../../core/buffer.hpp"
#include "../../exact/int128.hpp"
#include "../../exact/vertex.hpp"

namespace tf::exact {

/// Plane info for a convex face: 2D projection axes + indices of 3
/// non-collinear vertices that define the supporting plane.
struct face_plane_info {
  int ax0, ax1;
  std::size_t i0, i1, i2;
  bool valid;
};

/// Find 3 non-collinear vertices and compute 2D projection axes.
template <typename Index>
auto compute_face_plane(const tf::buffer<vertex<Index>> &face)
    -> face_plane_info {
  auto n = face.size();
  std::size_t id0 = 0, id1 = 1;
  while (id1 < n && face[id1].pt[0] == face[id0].pt[0] &&
         face[id1].pt[1] == face[id0].pt[1] &&
         face[id1].pt[2] == face[id0].pt[2])
    ++id1;
  if (id1 >= n)
    return {0, 1, 0, 0, 0, false};
  int64_t e0x = int64_t(face[id1].pt[0]) - int64_t(face[id0].pt[0]);
  int64_t e0y = int64_t(face[id1].pt[1]) - int64_t(face[id0].pt[1]);
  int64_t e0z = int64_t(face[id1].pt[2]) - int64_t(face[id0].pt[2]);
  int128 nx = 0, ny = 0, nz = 0;
  std::size_t id2 = id1 + 1;
  for (; id2 < n; ++id2) {
    int64_t e1x = int64_t(face[id2].pt[0]) - int64_t(face[id0].pt[0]);
    int64_t e1y = int64_t(face[id2].pt[1]) - int64_t(face[id0].pt[1]);
    int64_t e1z = int64_t(face[id2].pt[2]) - int64_t(face[id0].pt[2]);
    nx = int128(e0y) * e1z - int128(e0z) * e1y;
    ny = int128(e0z) * e1x - int128(e0x) * e1z;
    nz = int128(e0x) * e1y - int128(e0y) * e1x;
    if (nx != 0 || ny != 0 || nz != 0)
      break;
  }
  if (id2 >= n)
    return {0, 1, 0, 0, 0, false};
  if (nx < 0)
    nx = -nx;
  if (ny < 0)
    ny = -ny;
  if (nz < 0)
    nz = -nz;
  if (nz >= nx && nz >= ny)
    return {0, 1, id0, id1, id2, true};
  if (ny >= nx)
    return {0, 2, id0, id1, id2, true};
  return {1, 2, id0, id1, id2, true};
}

} // namespace tf::exact

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
#include "../../exact/meta.hpp"
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
template <typename Index, typename Int>
auto compute_face_plane(const tf::buffer<vertex<Index, Int>> &face)
    -> face_plane_info {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  auto n = face.size();
  std::size_t id0 = 0, id1 = 1;
  while (id1 < n && face[id1].pt[0] == face[id0].pt[0] &&
         face[id1].pt[1] == face[id0].pt[1] &&
         face[id1].pt[2] == face[id0].pt[2])
    ++id1;
  if (id1 >= n)
    return {0, 1, 0, 0, 0, false};
  T1 e0x = T1(face[id1].pt[0]) - T1(face[id0].pt[0]);
  T1 e0y = T1(face[id1].pt[1]) - T1(face[id0].pt[1]);
  T1 e0z = T1(face[id1].pt[2]) - T1(face[id0].pt[2]);
  T2 nx = 0, ny = 0, nz = 0;
  std::size_t id2 = id1 + 1;
  for (; id2 < n; ++id2) {
    T1 e1x = T1(face[id2].pt[0]) - T1(face[id0].pt[0]);
    T1 e1y = T1(face[id2].pt[1]) - T1(face[id0].pt[1]);
    T1 e1z = T1(face[id2].pt[2]) - T1(face[id0].pt[2]);
    nx = T2(e0y) * e1z - T2(e0z) * e1y;
    ny = T2(e0z) * e1x - T2(e0x) * e1z;
    nz = T2(e0x) * e1y - T2(e0y) * e1x;
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

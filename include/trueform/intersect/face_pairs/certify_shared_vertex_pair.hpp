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
#include "../../core/config/verify.hpp"
#include "../../exact/orient3d.hpp"
#include "../../exact/vertex.hpp"
#include "../classify/face_plane_info.hpp"

#include <cstddef>

namespace tf::intersect {

/// Whether triangles (p, a1, a2) and (p, b1, b2), p at `i` and `j`, meet in
/// p alone: one strictly on a side of the other's plane meets it only there;
/// when both straddle, their spans on the planes' common line run from p
/// the opposite way — and so meet in p alone — iff a1's sign against B's
/// plane equals b1's against A's, each plane standing on its corners in
/// face order.
template <typename Index, typename Int>
auto certify_shared_vertex_pair(tf::exact::vertex_range<Index, Int> face0,
                                const tf::exact::face_plane<Int> &fp0,
                                std::size_t i,
                                tf::exact::vertex_range<Index, Int> face1,
                                const tf::exact::face_plane<Int> &fp1,
                                std::size_t j) -> bool {
  if (!fp0.valid() || !fp1.valid())
    return false;
  const auto plane1 = tf::exact::face_orient3d_plane(fp1, face1);
  const auto &p = face0[i].pt;
  const auto &a1 = face0[(i + 1) % 3].pt;
  const auto &a2 = face0[(i + 2) % 3].pt;
  const int sa1 = tf::exact::orient3d_plane_sign(plane1, a1);
  const int sa2 = tf::exact::orient3d_plane_sign(plane1, a2);
  if (sa1 == sa2 && sa1 != 0)
    return true;
  const auto &b1 = face1[(j + 1) % 3].pt;
  const auto &b2 = face1[(j + 2) % 3].pt;
  const auto plane0 = tf::exact::face_orient3d_plane(fp0, face0);
  const int sb1 = tf::exact::orient3d_plane_sign(plane0, b1);
  const int sb2 = tf::exact::orient3d_plane_sign(plane0, b2);
  if (sb1 == sb2 && sb1 != 0)
    return true;
  if (sa1 * sa2 != -1 || sb1 * sb2 != -1)
    return false;
  if constexpr (tf::core::verification)
    tf::core::verify(tf::exact::orient3d_sign(p, b1, a1, a2) == sb1,
                     "a triangle's plane stands on its corners in face order");
  return sa1 == sb1;
}

} // namespace tf::intersect

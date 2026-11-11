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
#include "../../exact/orient3d.hpp"
#include "../../exact/vertex.hpp"
#include "../classify/face_plane_info.hpp"

#include <cstddef>

namespace tf::intersect {

/// Whether two triangles sharing the edge opposite `apex0` of the first meet
/// in that edge alone: when the apex stands off the second's plane the two
/// planes meet in the edge's line, which each triangle meets in the edge.
template <typename Index, typename Int>
auto certify_shared_edge_pair(tf::exact::vertex_range<Index, Int> face0,
                              std::size_t apex0,
                              tf::exact::vertex_range<Index, Int> face1,
                              const tf::exact::face_plane<Int> &fp1) -> bool {
  return fp1.valid() &&
         tf::exact::orient3d_plane_sign(
             tf::exact::face_orient3d_plane(fp1, face1), face0[apex0].pt) != 0;
}

} // namespace tf::intersect

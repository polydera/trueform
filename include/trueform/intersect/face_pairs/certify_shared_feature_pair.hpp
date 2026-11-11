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
#include "../../exact/vertex.hpp"
#include "../classify/face_plane_info.hpp"
#include "./certify_shared_edge_pair.hpp"
#include "./certify_shared_vertex_pair.hpp"

#include <cstddef>

namespace tf::intersect {

/// Whether two triangles of one form meet in exactly the vertex or the edge
/// their corner ids share, by that class's own certificate. `false` states
/// nothing: the pair goes to the general kernel.
template <typename Index, typename Indices0, typename Indices1, typename Int>
auto certify_shared_feature_pair(const Indices0 &ids0, const Indices1 &ids1,
                                 tf::exact::vertex_range<Index, Int> face0,
                                 const tf::exact::face_plane<Int> &fp0,
                                 tf::exact::vertex_range<Index, Int> face1,
                                 const tf::exact::face_plane<Int> &fp1)
    -> bool {
  if (ids0.size() != 3 || ids1.size() != 3)
    return false;
  int shared0 = 0, shared1 = 0;
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j)
      if (Index(ids0[i]) == Index(ids1[j])) {
        shared0 |= 1 << i;
        shared1 |= 1 << j;
      }
  const auto one_corner = [](int s) { return s == 1 || s == 2 || s == 4; };
  if (one_corner(shared0) && one_corner(shared1))
    return certify_shared_vertex_pair(face0, fp0, std::size_t(shared0 >> 1),
                                      face1, fp1, std::size_t(shared1 >> 1));
  if (one_corner(7 ^ shared0) && one_corner(7 ^ shared1))
    return certify_shared_edge_pair(face0, std::size_t((7 ^ shared0) >> 1),
                                    face1, fp1);
  return false;
}

} // namespace tf::intersect

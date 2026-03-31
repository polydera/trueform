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

#include "../../exact/orient3d.hpp"
#include <array>

namespace tf::exact {

/// Compare two coplanar intersection points along an edge.
/// face_a, face_b: the two faces (other mesh) that generated the points.
/// edge_vertex: one endpoint of the edge being sorted (this mesh).
/// Returns: -1 if face_a before face_b, +1 if after, 0 if undetermined.
///
/// Tests face_a's non-shared vertices against face_b's plane.
/// If they are on the same side as edge_vertex, face_a comes first.
template <typename Index, typename Int>
auto compare_coplanar_intersections(
    const std::array<vertex<Index, Int>, 3> &face_a,
    const std::array<vertex<Index, Int>, 3> &face_b,
    const vertex<Index, Int> &edge_vertex) -> int {
  // Find face_a's non-shared vertices
  int fa_non[3]{}, n_fa_non = 0;
  for (int i = 0; i < 3; ++i) {
    bool shared = false;
    for (int j = 0; j < 3; ++j)
      if (face_a[i].id == face_b[j].id) {
        shared = true;
        break;
      }
    if (!shared)
      fa_non[n_fa_non++] = i;
  }
  int n_shared = 3 - n_fa_non;

  auto side = [&](const vertex<Index, Int> &p) {
    return orient3d_sos(
        std::array<vertex<Index, Int>, 4>{face_b[0], face_b[1], face_b[2], p});
  };

  bool ev = side(edge_vertex);

  if (n_shared >= 2) {
    return side(face_a[fa_non[0]]) == ev ? -1 : 1;
  } else if (n_shared == 1) {
    bool o1 = side(face_a[fa_non[0]]);
    bool o2 = side(face_a[fa_non[1]]);
    if (o1 != o2)
      return 0;
    return o1 == ev ? -1 : 1;
  } else {
    bool o1 = side(face_a[fa_non[0]]);
    bool o2 = side(face_a[fa_non[1]]);
    bool o3 = side(face_a[fa_non[2]]);
    if (o1 != o2 || o1 != o3)
      return 0;
    return o1 == ev ? -1 : 1;
  }
}

} // namespace tf::exact

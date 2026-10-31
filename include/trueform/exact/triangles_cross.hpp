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

#include "./orient3d.hpp"
#include "./vertex.hpp"
#include <array>
#include <cstddef>

namespace tf::exact {

/// Whether triangles `t` and `u` cross: an edge of one has its ends strictly
/// on opposite sides of the other's plane and passes through the other, its
/// boundary included. A corner lying in the other's plane, and a coplanar
/// overlap, make no crossing.
template <typename Int>
auto triangles_cross(const std::array<pt3<Int>, 3> &t,
                     const std::array<pt3<Int>, 3> &u) -> bool {
  const auto pierces = [](const std::array<pt3<Int>, 3> &f,
                          const std::array<pt3<Int>, 3> &g) {
    std::array<int, 3> side{};
    for (std::size_t k = 0; k < 3; ++k)
      side[k] = orient3d_sign(f[0], f[1], f[2], g[k]);
    for (std::size_t k = 0; k < 3; ++k) {
      const auto j = (k + 1) % 3;
      if (side[k] * side[j] >= 0)
        continue;
      const int s0 = orient3d_sign(g[k], g[j], f[0], f[1]);
      const int s1 = orient3d_sign(g[k], g[j], f[1], f[2]);
      const int s2 = orient3d_sign(g[k], g[j], f[2], f[0]);
      if (!((s0 < 0 || s1 < 0 || s2 < 0) && (s0 > 0 || s1 > 0 || s2 > 0)))
        return true;
    }
    return false;
  };
  return pierces(t, u) || pierces(u, t);
}

} // namespace tf::exact

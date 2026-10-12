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

#include "../../core/point.hpp"
#include "../meta.hpp"
#include "../orient2d.hpp"
#include "../orient3d.hpp"
#include "../projection_axes.hpp"
#include "../vertex.hpp"

#include <cstddef>

namespace tf::exact::door {

/// Whether `u` stands exactly on the closed face its `corners` state —
/// interior, edge or corner. The face is read as the fan of its first
/// corner, and a fan triangle bounding no area holds nothing.
template <typename Int, typename Corners>
auto point_on_face(const tf::point<Int, 3> &u, const Corners &corners) -> bool {
  using T2 = typename tf::exact::meta<Int>::T2;
  const auto inside = [&u](const tf::point<Int, 3> &a,
                           const tf::point<Int, 3> &b,
                           const tf::point<Int, 3> &c) {
    if (tf::exact::orient3d_sign<Int>(a, b, c, u) != 0)
      return false;
    const auto [x, y] = tf::exact::projection_axes<Int>(a, b, c);
    const auto flat = [x = x, y = y](const tf::point<Int, 3> &p) {
      return tf::exact::pt2<Int>{p[std::size_t(x)], p[std::size_t(y)]};
    };
    const auto a2 = flat(a), b2 = flat(b), c2 = flat(c), u2 = flat(u);
    if (tf::exact::orient2d<Int>(a2, b2, c2) == T2(0))
      return false;
    const T2 s0 = tf::exact::orient2d<Int>(a2, b2, u2);
    const T2 s1 = tf::exact::orient2d<Int>(b2, c2, u2);
    const T2 s2 = tf::exact::orient2d<Int>(c2, a2, u2);
    return (s0 >= T2(0) && s1 >= T2(0) && s2 >= T2(0)) ||
           (s0 <= T2(0) && s1 <= T2(0) && s2 <= T2(0));
  };
  for (std::size_t k = 1; k + 1 < corners.size(); ++k)
    if (inside(corners[0], corners[k], corners[k + 1]))
      return true;
  return false;
}

} // namespace tf::exact::door

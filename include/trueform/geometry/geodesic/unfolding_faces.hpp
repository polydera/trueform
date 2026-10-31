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
#include "../../core/buffer.hpp"
#include "../../core/dot.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace tf::geometry {

/// What the unfolding reads of a triangle: the length of the edge leaving
/// each corner toward the next, and twice the area.
struct unfolding_face {
  std::array<double, 3> edges;
  double two_area;
};

/// The @ref tf::geometry::unfolding_face of one triangle, in double precision.
template <typename Polygon>
auto make_unfolding_face(const Polygon &polygon)
    -> tf::geometry::unfolding_face {
  const auto p0 = polygon[0].template as<double>();
  const auto p1 = polygon[1].template as<double>();
  const auto p2 = polygon[2].template as<double>();
  const auto a = p1 - p0;
  const auto b = p2 - p0;
  const double inner = tf::dot(a, b);
  return tf::geometry::unfolding_face{
      {a.length(), (p2 - p1).length(), b.length()},
      std::sqrt(std::max(0.0, a.length2() * b.length2() - inner * inner))};
}

/// The images one front reads, one slot per triangle: `images[t]` holds
/// triangle `t`'s once `formed[t]` says so. A slot is formed the first time a
/// pop reaches its triangle, so a front pays for the triangles it reaches.
struct unfolding_faces {
  tf::buffer<tf::geometry::unfolding_face> images;
  tf::buffer<char> formed;
};

} // namespace tf::geometry

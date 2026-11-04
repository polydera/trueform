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
#include "../../core/aabb_from.hpp"
#include "../../core/point.hpp"
#include "../../core/points.hpp"
#include "../../exact/pt_converter.hpp"
#include <cmath>
#include <cstddef>

namespace tf::tetmesh {

/// The converter onto the `Int` lattice at `bits` bits: the points' box
/// centred on the origin and scaled by the power of two that brings its
/// half extent within `2^(bits - 1)`, so a coordinate rounds once and the
/// lattice carries the input's precision and no more.
template <typename Int, typename Policy>
auto make_tet_entrance_converter(const tf::points<Policy> &points, int bits) {
  using RealT = tf::coordinate_type<Policy>;
  const auto box = tf::aabb_from(points);
  const auto center = box.center();
  RealT half = 0;
  for (std::size_t i = 0; i < 3; ++i)
    half = std::max(half, (box.max[i] - box.min[i]) / 2);
  int exponent = 0;
  if (half > 0) {
    std::frexp(half, &exponent);
  }
  const RealT scale = std::ldexp(RealT(1), bits - 1 - exponent);
  return tf::exact::pt_converter<Int, RealT, 3>{
      {scale, tf::point<RealT, 3>{center[0], center[1], center[2]}}};
}

} // namespace tf::tetmesh

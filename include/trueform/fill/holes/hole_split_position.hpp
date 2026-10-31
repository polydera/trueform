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
#include "../../core/points.hpp"
#include "../hole_split.hpp"
#include <cstddef>
#include <type_traits>

namespace tf::fill {

/// The point a split record names: `parameter / tf::hole_split_scale` along
/// the original edge from `v0` to `v1`, in the mesh's own coordinates. The
/// record is the split's identity and this is the one place its coordinate
/// is computed, so every tier that splits an edge states the same point.
template <typename PointsPolicy, typename Index>
auto hole_split_position(const tf::points<PointsPolicy> &points, Index v0,
                         Index v1, int parameter) {
  using RealT = std::decay_t<decltype(points[0][0])>;
  const auto &from = points[std::size_t(v0)];
  const auto &to = points[std::size_t(v1)];
  return tf::point<RealT, 3>(
      from + (to - from) * (RealT(parameter) / RealT(tf::hole_split_scale)));
}

} // namespace tf::fill

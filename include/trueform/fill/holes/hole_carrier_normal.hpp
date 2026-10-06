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
#include "../../core/faces.hpp"
#include "../../core/points.hpp"
#include "../../core/vector.hpp"
#include "./hole_metric.hpp"
#include <cstddef>

namespace tf::fill {

/// The doubled-area normal of a carrying triangle, read in its canonical
/// rotation, so one face states one value however its corners are named.
template <typename FacesPolicy, typename PointsPolicy, typename Index>
auto hole_carrier_normal(const tf::faces<FacesPolicy> &faces,
                         const tf::points<PointsPolicy> &points, Index face)
    -> tf::vector<double, 3> {
  const auto corners = faces[std::size_t(face)];
  const int first =
      tf::fill::hole_canonical_rotation(corners[0], corners[1], corners[2]);
  return tf::fill::hole_oriented_normal(
      points[std::size_t(corners[std::size_t(first)])].template as<double>(),
      points[std::size_t(corners[std::size_t((first + 1) % 3)])]
          .template as<double>(),
      points[std::size_t(corners[std::size_t((first + 2) % 3)])]
          .template as<double>());
}

} // namespace tf::fill

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
#include "../../core/points_buffer.hpp"
#include "../../core/polygons_buffer.hpp"
#include <cstddef>
#include <utility>

namespace tf::topology::cdt {

template <typename Coord, typename Triangulator>
auto materialize_unconstrained_delaunay(Triangulator &triangulator) {
  auto faces = triangulator.take_faces();
  tf::points_buffer<Coord, 2> points;
  if (faces.size() != 0) {
    points.allocate(triangulator.n_unique_points());
    for (std::size_t i = 0; i < points.size(); ++i) {
      const auto source = triangulator.converted_unique_point(i);
      auto destination = points[i];
      destination[0] = source[0];
      destination[1] = source[1];
    }
  }
  return tf::make_polygons_buffer(std::move(faces), std::move(points));
}

} // namespace tf::topology::cdt

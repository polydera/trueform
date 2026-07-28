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
#include "../../core/point.hpp"
#include "../../intersect/graph/vertex.hpp"
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int, typename IntersectionPoints,
          typename GetMeshPoint>
auto region_triangulation_point(
    Index n_intersection_points,
    const tf::buffer<tf::point<Int, 3>> &extra_points, Index tag,
    const tf::intersect::graph::vertex<Index> &vertex,
    const IntersectionPoints &intersection_points,
    const GetMeshPoint &get_mesh_point) -> tf::point<Int, 3> {
  if (vertex.source == tf::intersect::graph::vertex_source::created) {
    if (vertex.id >= n_intersection_points)
      return extra_points[std::size_t(vertex.id - n_intersection_points)];
    return intersection_points[vertex.id];
  }
  return get_mesh_point(tag, vertex.id);
}

} // namespace tf::cut

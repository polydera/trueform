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
#include "../../core/index_map.hpp"
#include "../../core/points.hpp"
#include "../../reindex/return_index_map.hpp"
#include "../unconstrained_delaunay_triangulator.hpp"
#include "./delaunay_execution_policy.hpp"
#include "./delaunay_vertex_policy.hpp"
#include "./materialize_unconstrained_delaunay.hpp"
#include <utility>

namespace tf::topology::cdt {

/// Build and materialize the compact point-only tier. Public `make_cdt`
/// overloads own type deduction; this operation owns its output assembly.
template <bool ReturnIndexMap, typename Index, typename Coord, typename Int,
          typename PointsPolicy>
auto make_unconstrained_delaunay(const tf::points<PointsPolicy> &points) {
  using VertexPolicy = tf::topology::cdt::compact_topology_vertex_policy;
  tf::unconstrained_delaunay_triangulator<
      Index, Coord, Int, VertexPolicy,
      tf::topology::cdt::parallel_delaunay_execution_policy>
      triangulator;

  if constexpr (ReturnIndexMap) {
    triangulator.build(points, tf::return_index_map);
    auto point_index_map = triangulator.take_index_map();
    auto output = tf::topology::cdt::materialize_unconstrained_delaunay<Coord>(
        triangulator);
    // An empty polygon buffer carries no points, so a retained map would name
    // output slots that do not exist.
    if (output.points_buffer().size() == 0)
      return std::make_pair(std::move(output), tf::index_map_buffer<Index>{});
    return std::make_pair(std::move(output), std::move(point_index_map));
  } else {
    triangulator.build(points);
    return tf::topology::cdt::materialize_unconstrained_delaunay<Coord>(
        triangulator);
  }
}

} // namespace tf::topology::cdt

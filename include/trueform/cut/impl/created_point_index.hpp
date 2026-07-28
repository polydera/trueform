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
#include "tbb/parallel_sort.h"
#include <algorithm>
#include <cstddef>

namespace tf::cut {

/// @ingroup cut
/// @brief Sorted lookup from a lattice coordinate to the created point
///        already occupying it.
///
/// One lattice coordinate is one created vertex. The intersection graph
/// enforces this for its own points with
/// @ref tf::intersect::dedup_coincident_points; crossings the recovery wave
/// materializes are the created identity nobody dedups, and a coordinate
/// holding two of them splits the constraints through it at two different
/// vertices.
///
/// Ordered by coordinate with ties broken by id, so a duplicate group's
/// survivor is its smallest id — the tie-break
/// @ref tf::make_unique_index_map uses, and deterministic across runs.
template <typename Index, typename Int> struct created_point_record {
  tf::point<Int, 3> position;
  Index id;
};

template <typename Index, typename Int>
auto build_created_point_index(
    const tf::buffer<tf::point<Int, 3>> &points,
    tf::buffer<tf::cut::created_point_record<Index, Int>> &order) -> void {
  order.allocate(points.size());
  if (points.size() == 0)
    return;
  for (std::size_t i = 0; i < points.size(); ++i)
    order[i] = {points[i], Index(i)};
  // Compact records move by value; sorting ids and comparing them through a
  // lookup into the positions gathers twice per comparison.
  tbb::parallel_sort(
      order, [](const tf::cut::created_point_record<Index, Int> &a,
                const tf::cut::created_point_record<Index, Int> &b) {
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
          if (a.position[coordinate] != b.position[coordinate])
            return a.position[coordinate] < b.position[coordinate];
        }
        return a.id < b.id;
      });
}

/// @ingroup cut
/// @brief The created point at `query`, or `-1` when the coordinate is
///        unoccupied. Expects the order @ref
///        tf::cut::build_created_point_index leaves behind.
template <typename Index, typename Int>
auto find_created_point(
    const tf::buffer<tf::cut::created_point_record<Index, Int>> &order,
    const tf::point<Int, 3> &query) -> Index {
  auto less = [](const tf::cut::created_point_record<Index, Int> &a,
                 const tf::point<Int, 3> &q) {
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      if (a.position[coordinate] != q[coordinate])
        return a.position[coordinate] < q[coordinate];
    }
    return false;
  };
  auto found = std::lower_bound(order.begin(), order.end(), query, less);
  if (found == order.end())
    return Index(-1);
  for (int coordinate = 0; coordinate < 3; ++coordinate)
    if (found->position[coordinate] != query[coordinate])
      return Index(-1);
  return found->id;
}

} // namespace tf::cut

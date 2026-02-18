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

#include "../core/algorithm/reduce.hpp"
#include "../core/polygons.hpp"
#include "../core/views/mapped_range.hpp"

namespace tf {

/// @ingroup topology_analysis
/// @brief Computes the Euler characteristic of a polygon mesh.
///
/// The Euler characteristic is defined as V - E + F, where V is the number
/// of vertices, E is the number of unique edges, and F is the number of
/// faces. Edges are counted by iterating all face edges and counting only
/// those where the first vertex index is less than the second, avoiding
/// double-counting of shared edges on manifold meshes.
///
/// Requires a closed manifold mesh. Behavior is undefined otherwise.
///
/// @tparam Policy The polygons policy.
/// @param polygons The polygon collection.
/// @return The Euler characteristic (V - E + F).
template <typename Policy>
auto euler_characteristic(const tf::polygons<Policy> &polygons) -> int {
  int V = polygons.points().size();
  int F = polygons.faces().size();

  int E = tf::reduce(
      tf::make_mapped_range(
          polygons,
          [](const auto &polygon) {
            int count = 0;
            auto size = polygon.size();
            decltype(size) prev = size - 1;
            for (decltype(size) i = 0; i < size; prev = i++) {
              if (polygon.indices()[prev] < polygon.indices()[i])
                count++;
            }
            return count;
          }),
      std::plus<>{}, int{0}, tf::checked);

  return V - E + F;
}

} // namespace tf

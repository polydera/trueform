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
#include "../../core/polygon.hpp"
#include "../../core/polygons.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include <cstddef>

namespace tf::geometry {

/// The one-face mesh a single loop states: the loop's own points, and the one
/// face that names them in order.
///
/// A loop that repeats its first POINT at the end draws the same loop one
/// corner shorter, so the repeat is named by nothing. It stays in the table
/// where the caller put it — the table is the caller's statement of its own
/// points, and a face is what names them.
template <typename Index, std::size_t Dims, typename Policy>
auto make_loop_polygons(const tf::polygon<Dims, Policy> &polygon) {
  const auto n_points = polygon.size();
  const auto n_corners =
      n_points -
      std::size_t(n_points > 1 && polygon[0] == polygon[n_points - 1]);
  // an empty sequence yields no block whatever the block is, and a loop with
  // corners is one block of its own length
  return tf::make_polygons(
      tf::make_blocked_range(tf::make_sequence_range(Index(n_corners)),
                             n_corners == 0 ? std::size_t(1) : n_corners),
      polygon);
}

} // namespace tf::geometry

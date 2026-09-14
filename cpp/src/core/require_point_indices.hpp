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

#include "trueform/core/algorithm/parallel_contains.hpp"
#include "trueform/core/checked.hpp"

#include <stdexcept>
#include <string>
#include <type_traits>

namespace tf::cpp::carrier {

/// @brief Refuse corners that name a point their reading does not have.
///
/// The one producer of that fact for every carrier whose primitives are point
/// indices and every entry handed some: a mesh's faces, an edge mesh's edges,
/// and the arrays a caller triangulates without owning them. The corners are
/// the carrier, so the scan runs at every one of them at once.
///
/// The caller adds what only it knows: how many points a corner may name, and
/// the door it is refused at.
template <typename Range>
auto require_point_indices(const Range &indices, int number_of_points,
                           const char *name) -> void {
  using index_type = std::decay_t<decltype(*indices.begin())>;
  const auto limit = static_cast<index_type>(number_of_points);
  const auto refused = tf::parallel_contains(
      indices,
      [limit](index_type index) {
        return index < index_type{0} || index >= limit;
      },
      tf::checked);
  if (refused)
    throw std::out_of_range(std::string(name) +
                            ": point index is out of bounds");
}

} // namespace tf::cpp::carrier

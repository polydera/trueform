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
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// Per created point id of the records `records` — the ids past
/// `n_vertex_points()` — the smallest pair of input faces (`parents` of the
/// traced triangles `cut`) a record names it for, into `pairs` by id; `-1`
/// for an id no record of two triangles names. The pairs a drain's authority
/// mints its points for (@ref tet_identity_authority::name_positions).
/// O(records).
template <typename Index, typename Records, typename Cut, typename Parents>
auto state_tet_point_keys(const Records &records, const Cut &cut,
                          const Parents &parents,
                          tf::buffer<std::array<Index, 2>> &pairs) -> void {
  const auto n_vertex = std::size_t(records.n_vertex_points());
  const auto n_ids = std::size_t(records.n_points());
  pairs.allocate(n_ids - n_vertex);
  std::fill(pairs.begin(), pairs.end(),
            std::array<Index, 2>{Index(-1), Index(-1)});
  for (const auto &record : records.flat_intersections()) {
    if (std::size_t(record.id) < n_vertex ||
        record.object == record.object_other)
      continue;
    const Index a =
        Index(parents[std::size_t(cut[std::size_t(record.object)])]);
    const Index b =
        Index(parents[std::size_t(cut[std::size_t(record.object_other)])]);
    const std::array<Index, 2> key{std::min(a, b), std::max(a, b)};
    auto &slot = pairs[std::size_t(record.id) - n_vertex];
    if (slot[0] < 0 || key < slot)
      slot = key;
  }
}

} // namespace tf::tetmesh

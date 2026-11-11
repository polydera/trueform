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
#include "../../core/offset_block_buffer.hpp"
#include "../../topology/cdt/dt3/recovery/tet_face_status.hpp"
#include <array>
#include <cstddef>

namespace tf::tetmesh {

/// The published facets, densely numbered: every facet `status` does not
/// call retired, in order, its triangle run of `facets` and their `parents`
/// into `kept_facets` and `kept_parents`, and its block of `children` into
/// `kept_children`. The retired ones are rows of the wave alone.
template <typename Index, typename Status>
auto compact_tet_facets(
    const tf::buffer<Index> &facets, const tf::buffer<Index> &parents,
    const Status &status,
    const tf::offset_block_buffer<Index, std::array<Index, 3>> &children,
    tf::buffer<Index> &kept_facets, tf::buffer<Index> &kept_parents,
    tf::offset_block_buffer<Index, std::array<Index, 3>> &kept_children)
    -> void {
  namespace dt3 = tf::topology::cdt::dt3;
  const std::size_t n = facets.size() ? facets.size() - 1 : 0;
  kept_facets.clear();
  kept_parents.clear();
  kept_facets.push_back(Index(0));
  auto &offsets = kept_children.offsets_buffer();
  auto &data = kept_children.data_buffer();
  offsets.clear();
  data.clear();
  offsets.push_back(Index(0));
  // one ordered pass: the kept runs are copies the memory bounds, which an
  // offset-block generation over the kept facets does not beat
  for (std::size_t f = 0; f < n; ++f) {
    if (status[f] == dt3::tet_face_status::retired)
      continue;
    for (Index t = facets[f]; t < facets[f + 1]; ++t)
      kept_parents.push_back(parents[std::size_t(t)]);
    kept_facets.push_back(Index(kept_parents.size()));
    for (const auto &triangle : children[f])
      data.push_back(triangle);
    offsets.push_back(Index(data.size()));
  }
}

} // namespace tf::tetmesh

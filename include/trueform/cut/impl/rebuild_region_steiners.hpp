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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/buffer.hpp"
#include "../../core/offset_block_buffer.hpp"
#include <array>
#include <cstddef>

namespace tf::cut {

/// @ingroup cut
/// @brief The interior points every region holds after a refinement pass.
///
/// A region's interior points are chosen once and then carried through
/// every triangulation of it — only its constraints are cut afterwards. A
/// later pass at a higher quality adds to a region's block; it never takes
/// away, because a point already placed is already in the table every
/// carrier reads.
///
/// `processed` is {region, count} for the regions this pass looked at, in
/// the order their points were appended; their ids run contiguously from
/// `base`.
template <typename Index>
auto rebuild_region_steiners(
    std::size_t n_regions, const tf::buffer<std::array<Index, 2>> &processed,
    Index base, tf::offset_block_buffer<Index, Index> &steiners) -> void {
  tf::buffer<Index> counts;
  tf::buffer<Index> carried;
  counts.allocate(n_regions);
  carried.allocate(n_regions);
  const bool had_previous = steiners.size() == n_regions;
  for (std::size_t region = 0; region < n_regions; ++region) {
    carried[region] =
        had_previous ? Index(steiners[region].size()) : Index(0);
    counts[region] = carried[region];
  }
  for (const auto &entry : processed)
    counts[std::size_t(entry[0])] += entry[1];

  tf::buffer<Index> offsets;
  offsets.allocate(n_regions + 1);
  offsets[0] = 0;
  for (std::size_t region = 0; region < n_regions; ++region)
    offsets[region + 1] = offsets[region] + counts[region];

  tf::buffer<Index> data;
  data.allocate(std::size_t(offsets[n_regions]));
  if (had_previous)
    for (std::size_t region = 0; region < n_regions; ++region)
      for (std::size_t i = 0; i < std::size_t(carried[region]); ++i)
        data[std::size_t(offsets[region]) + i] = steiners[region][i];

  Index appended = 0;
  for (const auto &entry : processed) {
    const std::size_t at = std::size_t(offsets[std::size_t(entry[0])]) +
                           std::size_t(carried[std::size_t(entry[0])]);
    for (Index i = 0; i < entry[1]; ++i)
      data[at + std::size_t(i)] = base + appended + i;
    appended += entry[1];
  }

  steiners.offsets_buffer() = std::move(offsets);
  steiners.data_buffer() = std::move(data);
}

} // namespace tf::cut

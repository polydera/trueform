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
#include <array>
#include <cstddef>
#include <utility>

namespace tf::cut {

template <typename Index, typename Triangle>
auto splice_region_triangulations(
    const tf::buffer<Index> &dirty,
    const tf::buffer<std::array<Index, 2>> &replacement_ranges,
    const tf::buffer<Triangle> &replacement_triangles,
    tf::buffer<std::array<Index, 2>> &ranges,
    tf::buffer<Triangle> &triangles) -> void {
  tf::buffer<Triangle> merged;
  tf::buffer<std::array<Index, 2>> merged_ranges;
  merged_ranges.allocate(ranges.size());
  std::size_t dirty_index = 0;
  Index offset = 0;
  for (std::size_t loop = 0; loop < ranges.size(); ++loop) {
    const bool replaced =
        dirty_index < dirty.size() && dirty[dirty_index] == Index(loop);
    const auto range =
        replaced ? replacement_ranges[dirty_index] : ranges[loop];
    const auto &source = replaced ? replacement_triangles : triangles;
    for (Index triangle = range[0]; triangle < range[1]; ++triangle)
      merged.push_back(source[std::size_t(triangle)]);
    merged_ranges[loop] = {offset, offset + (range[1] - range[0])};
    offset += range[1] - range[0];
    dirty_index += replaced;
  }
  triangles = std::move(merged);
  ranges = std::move(merged_ranges);
}

} // namespace tf::cut

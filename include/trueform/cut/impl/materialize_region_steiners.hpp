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
#include "../../core/memory.hpp"
#include "../../core/point.hpp"
#include "../detail/region_triangulation_types.hpp"
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int>
auto materialize_region_steiners(
    std::size_t n_regions,
    const tf::buffer<std::array<Index, 2>> &steiner_counts,
    const tf::buffer<tf::point<Int, 3>> &steiner_points,
    tf::buffer<tf::point<Int, 3>> &extra_points)
    -> tf::cut::detail::materialized_region_steiners<Index> {
  tf::cut::detail::materialized_region_steiners<Index> out;
  out.offsets.allocate(n_regions + 1);
  tf::parallel_fill(out.offsets, Index(0));
  for (const auto &count : steiner_counts)
    out.offsets[std::size_t(count[0]) + 1] = count[1];
  for (std::size_t region = 0; region < n_regions; ++region)
    out.offsets[region + 1] += out.offsets[region];
  out.base = Index(extra_points.size());
  tf::core::append(steiner_points, extra_points);
  return out;
}

} // namespace tf::cut

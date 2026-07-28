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
#include "../../core/point.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_types.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int>
auto inject_region_steiners(
    Index region,
    const tf::offset_block_buffer<Index, Index> &region_steiners,
    Index n_intersection_points,
    const tf::buffer<tf::point<Int, 3>> &extra_points,
    tf::cut::detail::region_triangulation_workspace<Index, Int> &workspace)
    -> void {
  if (region_steiners.size() == 0)
    return;
  for (auto id : region_steiners[std::size_t(region)]) {
    const auto &point =
        extra_points[std::size_t(id - n_intersection_points)];
    workspace.ihm.kept_ids().push_back({
        tf::intersect::graph::vertex_source::created,
        id,
        {0, tf::topo_type::face}});
    workspace.pts.push_back(
        {point[workspace.ax0], point[workspace.ax1]});
    workspace.pos_set.push_back(0);
  }
}

} // namespace tf::cut

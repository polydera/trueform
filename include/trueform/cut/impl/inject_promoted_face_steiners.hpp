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
#include "../../intersect/graph/vertex.hpp"
#include "../detail/region_triangulation_types.hpp"
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int>
auto inject_promoted_face_steiners(
    Index begin, Index end,
    const tf::buffer<std::array<Index, 3>> &promoted_steiners,
    Index n_intersection_points,
    const tf::buffer<tf::point<Int, 3>> &extra_points,
    tf::cut::detail::region_triangulation_workspace<Index, Int> &workspace)
    -> void {
  for (auto steiner =
           promoted_steiners.begin() + std::ptrdiff_t(begin);
       steiner != promoted_steiners.begin() + std::ptrdiff_t(end);
       ++steiner) {
    const Index id = (*steiner)[2];
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

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
#include "../../core/points.hpp"
#include "../hole_split.hpp"
#include "./hole_rim.hpp"
#include "./hole_table.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::fill {

template <typename PointsPolicy, typename Index, typename Int, typename RealT>
auto expand_hole_rim_splits(
    const tf::points<PointsPolicy> &points,
    const tf::fill::hole_rim<Index, Int, RealT> &rim,
    tf::fill::hole_table_scratch<Index, Int, RealT> &scratch) -> void {
  const Index n = Index(rim.size());
  scratch.next_site_of_position.clear();
  scratch.sites.clear();
  scratch.positions.clear();
  scratch.rim_edges.clear();
  scratch.corners.clear();
  std::size_t wave = 0;
  for (Index k = 0; k < n; ++k) {
    scratch.sites.push_back(rim.sites[std::size_t(k)]);
    scratch.next_site_of_position.push_back(
        scratch.site_of_position[std::size_t(k)]);
    scratch.positions.push_back(rim.positions[std::size_t(k)]);
    scratch.corners.push_back(rim.corners[std::size_t(k)]);
    const auto edge = rim.edges[std::size_t(k)];
    if (wave == scratch.edge_query.missing.size() ||
        scratch.edge_query.missing[wave] != k) {
      scratch.rim_edges.push_back(edge);
      continue;
    }
    ++wave;
    const auto parameter = std::uint8_t((int(edge.t0) + int(edge.t1)) / 2);
    auto near_side = edge;
    near_side.t1 = parameter;
    auto far_side = edge;
    far_side.t0 = parameter;
    scratch.rim_edges.push_back(near_side);
    scratch.rim_edges.push_back(far_side);
    scratch.next_site_of_position.push_back(
        Index(scratch.dt.n_sites() + wave - 1));
    scratch.sites.push_back(scratch.appended_sites[wave - 1]);
    scratch.corners.push_back(Index(-1));
    const auto &from = points[std::size_t(edge.v0)];
    const auto &to = points[std::size_t(edge.v1)];
    scratch.positions.push_back(
        from + (to - from) * (RealT(parameter) / RealT(tf::hole_split_scale)));
  }
}

} // namespace tf::fill

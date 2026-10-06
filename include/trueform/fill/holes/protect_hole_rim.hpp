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
#include "../../core/range.hpp"
#include "../../topology/tetrahedralization_refusal.hpp"
#include "../hole_fill_status.hpp"
#include "./expand_hole_rim_splits.hpp"
#include "./hole_rim.hpp"
#include "./hole_table.hpp"
#include "./prepare_hole_rim_splits.hpp"
#include "./state_missing_rim_edges.hpp"
#include <cstddef>
#include <utility>

namespace tf::fill {

/// Recover the rim's edges by exact dyadic subdivision and retained insertion.
/// Coincident split positions refuse a non-simple rim; host positions are
/// interpolated on the original edge, preserving its face and parameter.
template <typename PointsPolicy, typename Index, typename Int, typename RealT>
auto protect_hole_rim(const tf::points<PointsPolicy> &points,
                      tf::fill::hole_table_scratch<Index, Int, RealT> &scratch,
                      tf::fill::hole_rim<Index, Int, RealT> &rim,
                      Index &offending) -> tf::hole_fill_status {
  if (!scratch.dt.build(tf::make_range(rim.sites)))
    return scratch.dt.refusal() ==
                   tf::tetrahedralization_refusal::index_capacity
               ? tf::hole_fill_status::refused_resource
               : tf::hole_fill_status::refused_table;
  scratch.site_of_position.allocate(rim.size());
  for (std::size_t k = 0; k < rim.size(); ++k)
    scratch.site_of_position[k] = scratch.dt.index_map().f()[k];
  for (;;) {
    tf::fill::state_missing_rim_edges(scratch.dt, scratch.site_of_position,
                                      scratch.edge_query);
    if (!scratch.edge_query.missing.size()) {
      scratch.position_of_site.allocate(scratch.dt.n_sites());
      for (std::size_t k = 0; k < rim.size(); ++k)
        scratch.position_of_site[std::size_t(scratch.site_of_position[k])] =
            Index(k);
      return tf::hole_fill_status::filled;
    }

    const auto status =
        tf::fill::prepare_hole_rim_splits(rim, scratch, offending);
    if (status != tf::hole_fill_status::filled)
      return status;
    tf::fill::expand_hole_rim_splits(points, rim, scratch);
    if (!scratch.dt.append_sites(tf::make_range(scratch.appended_sites)))
      return tf::hole_fill_status::refused_resource;
    std::swap(scratch.site_of_position, scratch.next_site_of_position);
    std::swap(rim.sites, scratch.sites);
    std::swap(rim.positions, scratch.positions);
    std::swap(rim.corners, scratch.corners);
    std::swap(rim.edges, scratch.rim_edges);
  }
}

} // namespace tf::fill

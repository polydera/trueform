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
#include "../../core/range.hpp"
#include "../../core/views/take.hpp"
#include "../../topology/cdt/dt3/find_tet_site_coincidence.hpp"
#include "../../topology/cdt/dt3/propose_tet_edge_midpoints.hpp"
#include "../hole_fill_status.hpp"
#include "./hole_rim.hpp"
#include "./hole_rim_site_edges.hpp"
#include "./hole_table.hpp"
#include <algorithm>
#include <cstddef>
#include <limits>

namespace tf::fill {

template <typename Index, typename Int, typename RealT>
auto prepare_hole_rim_splits(
    const tf::fill::hole_rim<Index, Int, RealT> &rim,
    tf::fill::hole_table_scratch<Index, Int, RealT> &scratch,
    Index &offending) -> tf::hole_fill_status {
  const auto &missing = scratch.edge_query.missing;
  const auto maximum = std::size_t(std::numeric_limits<Index>::max());
  const auto first_name = scratch.dt.next_name();
  const auto available_names =
      first_name > maximum ? std::size_t(0) : maximum - first_name + 1;
  const auto available =
      std::min(maximum - scratch.dt.n_sites(), available_names);
  std::size_t count = 0;
  auto status = tf::hole_fill_status::filled;
  for (; count < missing.size(); ++count) {
    const auto edge = rim.edges[std::size_t(missing[count])];
    if ((int(edge.t0) + int(edge.t1)) % 2) {
      status = tf::hole_fill_status::refused_protection_depth;
      break;
    }
    if (count == available) {
      status = tf::hole_fill_status::refused_resource;
      break;
    }
  }
  if (!count) {
    scratch.appended_sites.clear();
    if (missing.size())
      offending = missing[0];
    return status;
  }
  tf::topology::cdt::dt3::propose_tet_edge_midpoints(
      scratch.dt, tf::fill::hole_rim_site_edges(scratch.site_of_position),
      tf::take(missing, count), Index(first_name), scratch.appended_sites);
  const auto collision = tf::topology::cdt::dt3::find_tet_site_coincidence(
      scratch.dt.sites(), tf::make_range(scratch.appended_sites),
      scratch.site_candidates);
  if (collision < count) {
    offending = missing[collision];
    return tf::hole_fill_status::refused_invalid_rim;
  }
  if (count < missing.size())
    offending = missing[count];
  return status;
}

} // namespace tf::fill

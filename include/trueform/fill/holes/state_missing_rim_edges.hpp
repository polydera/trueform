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
#include "../../topology/cdt/dt3/state_missing_tet_edges.hpp"
#include "../../topology/cdt/dt3/tet_edge_query_workspace.hpp"
#include "./hole_rim_site_edges.hpp"

namespace tf::fill {

template <typename DT, typename Index>
auto state_missing_rim_edges(
    const DT &dt, const tf::buffer<Index> &site_of_position,
    tf::topology::cdt::dt3::tet_edge_query_workspace<Index> &work) -> void {
  tf::topology::cdt::dt3::state_missing_tet_edges(
      dt, tf::fill::hole_rim_site_edges(site_of_position), work);
}

} // namespace tf::fill

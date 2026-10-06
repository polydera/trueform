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
#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "./state_tet_site_cells.hpp"
#include "./tet_contains_edge.hpp"
#include "./tet_edge_query_workspace.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Edges name published site slots. Missing entries retain input edge order.
template <typename DT, typename Edges, typename Index>
auto state_missing_tet_edges(const DT &dt, const Edges &edges,
                             tet_edge_query_workspace<Index> &work) -> void {
  state_tet_site_cells(dt, work.incident);
  work.present.allocate(edges.size());
  tf::parallel_for_each(
      tf::make_sequence_range(edges.size()),
      [&](std::size_t i, tet_edge_query_scratch<Index> &local) {
        const auto e = edges[i];
        work.present[i] = tet_contains_edge(
            dt, e[0], e[1], work.incident[std::size_t(e[0])], local);
      },
      tet_edge_query_scratch<Index>{}, tf::checked);
  work.missing.clear();
  for (std::size_t i = 0; i < edges.size(); ++i)
    if (!work.present[i])
      work.missing.push_back(Index(i));
}

} // namespace tf::topology::cdt::dt3

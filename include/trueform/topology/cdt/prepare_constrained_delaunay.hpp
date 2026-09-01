/*
 * Copyright (c) 2025 XLAB
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
#include "../../core/edges.hpp"
#include "../../core/point.hpp"
#include "../../core/points.hpp"
#include "../../core/range.hpp"
#include "./finalize_delaunay_index_map.hpp"
#include "./prepare_delaunay_sites.hpp"
#include <cstddef>

namespace tf::topology::cdt {

/// Convert, Morton-sort, and exact-weld into the shared site authority once.
/// Retained point identity and remapped constraints are materialized from that
/// carrier before the Delaunay build consumes it.
template <typename Owner, typename PointsPolicy, typename EdgesPolicy,
          typename Iterator, std::size_t N>
auto prepare_constrained_delaunay(Owner &owner,
                                  const tf::points<PointsPolicy> &points,
                                  const tf::edges<EdgesPolicy> &edges,
                                  const tf::range<Iterator, N> &is_boundary)
    -> void {
  using Index = typename Owner::index_type;
  using ExecutionPolicy = typename Owner::execution_policy;
  prepare_delaunay_sites<true, ExecutionPolicy::parallel, true>(
      owner, points, owner._converter);
  finalize_delaunay_index_map(owner, points.size());
  owner._has_prepared_input_index_map = true;

  owner._points.reserve(owner._sites.size());
  for (std::size_t output = 0; output < owner._sites.size(); ++output) {
    auto &site = owner._sites[output];
    owner._points.push_back(tf::make_point(site.x, site.y));
    site.output = static_cast<Index>(output);
  }

  const Index n_edges = static_cast<Index>(edges.size());
  owner._final_constraint_edges.reserve(n_edges);
  owner._final_is_boundary.reserve(n_edges);
  owner._constraint_input_ids.clear();
  if (owner._track_constraint_provenance)
    owner._constraint_input_ids.reserve(n_edges);
  auto boundary = is_boundary.begin();
  for (Index i = 0; i < n_edges; ++i, ++boundary) {
    const Index first = owner._index_map.f()[edges[i][0]];
    const Index second = owner._index_map.f()[edges[i][1]];
    if (first == second)
      continue;
    owner._final_constraint_edges.push_back({first, second});
    owner._final_is_boundary.push_back(*boundary ? 1 : 0);
    if (owner._track_constraint_provenance)
      owner._constraint_input_ids.push_back(i);
  }
}

} // namespace tf::topology::cdt

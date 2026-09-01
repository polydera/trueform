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
#include "../../core/points.hpp"
#include "../../exact_coordinate_converter.hpp"
#include "./build_delaunay_topology.hpp"
#include "./clear_unconstrained_delaunay.hpp"
#include "./finalize_delaunay_index_map.hpp"
#include "./materialize_unconstrained_delaunay_faces.hpp"
#include "./prepare_delaunay_sites.hpp"

namespace tf::topology::cdt {

/// Build the compact point-only Delaunay carriers. The vertex policy selects
/// only the IDs written to faces; every phase operates on the same site and
/// edge arrays.
template <bool BuildIndexMap, typename ExecutionPolicy, typename Owner,
          typename PointsPolicy>
auto build_unconstrained_delaunay(Owner &owner,
                                  const tf::points<PointsPolicy> &points)
    -> bool {
  using Index = typename Owner::index_type;
  using Coord = typename Owner::coord_type;
  using Int = typename Owner::int_type;

  tf::topology::cdt::clear_unconstrained_delaunay(owner);
  // The reserve formula also bounds every site and dart identity in Index.
  if (points.size() < 3 || points.size() > Owner::k_max_topology_sites)
    return false;

  owner._converter = tf::make_exact_coordinate_converter<Int, Coord>(points);
  tf::topology::cdt::prepare_delaunay_sites<true, ExecutionPolicy::parallel,
                                            BuildIndexMap>(owner, points,
                                                           owner._converter);
  if (owner._sites.size() < 3) {
    if constexpr (BuildIndexMap)
      tf::topology::cdt::finalize_delaunay_index_map(owner, points.size());
    return true;
  }

  owner._edges.reserve(owner._sites.size() * Owner::k_darts_per_site_reserve);
  const auto result =
      tf::topology::cdt::build_delaunay_topology<ExecutionPolicy>(owner,
                                                                  Owner::none);
  if constexpr (BuildIndexMap)
    tf::topology::cdt::finalize_delaunay_index_map(owner, points.size());
  tf::topology::cdt::materialize_unconstrained_delaunay_faces<ExecutionPolicy>(
      owner, result.outer_edge ^ Index(1));
  return true;
}

} // namespace tf::topology::cdt

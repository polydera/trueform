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
#include "../../core/points.hpp"
#include "../../core/range.hpp"
#include "../../exact_coordinate_converter.hpp"
#include "../cdt_region_mode.hpp"
#include "./build_constrained_delaunay_from_prepared_constraints.hpp"
#include "./clear_constrained_delaunay.hpp"
#include "./prepare_constrained_delaunay.hpp"
#include <cstddef>
#include <limits>

namespace tf::topology::cdt {

/// Convert and prepare the caller's arrays, recover constraints, and emit
/// triangle vertices without building adjacency or region labels.
template <typename Owner, typename PointsPolicy, typename EdgesPolicy,
          typename Iterator, std::size_t N>
auto build_constrained_delaunay_triangulation(
    Owner &owner, const tf::points<PointsPolicy> &points,
    const tf::edges<EdgesPolicy> &edges,
    const tf::range<Iterator, N> &is_boundary, bool split_constraints,
    tf::cdt_region_mode mode) -> bool {
  using Index = typename Owner::index_type;
  using Coord = typename Owner::coord_type;
  using Int = typename Owner::int_type;

  clear_constrained_delaunay(owner);
  if (points.size() > Owner::k_max_topology_sites ||
      edges.size() > std::size_t(std::numeric_limits<Index>::max()) ||
      is_boundary.size() < edges.size())
    return false;

  owner._converter = tf::make_exact_coordinate_converter<Int, Coord>(points);

  // The no-split pass must still refuse on a crossing so the recovery wave
  // fires; only the wave's retry resolves incrementally and records the
  // crossing provenance it creates.
  owner._allow_crossing_splits = split_constraints;
  owner._track_constraint_provenance =
      owner._always_track_constraint_provenance || split_constraints ||
      mode == tf::cdt_region_mode::components;

  prepare_constrained_delaunay(owner, points, edges, is_boundary);
  return build_constrained_delaunay_from_prepared_constraints(owner);
}

} // namespace tf::topology::cdt

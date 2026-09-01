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
#include "../../core/algorithm/compose_index_maps.hpp"
#include "../../core/edges.hpp"
#include "../../core/points.hpp"
#include "../../core/polygons.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../reindex/by_mask.hpp"
#include "../../reindex/return_index_map.hpp"
#include "../cdt_region_mode.hpp"
#include "../constrained_delaunay_triangulator.hpp"
#include "./delaunay_execution_policy.hpp"
#include <tuple>
#include <utility>

namespace tf::topology::cdt {

/// Build the retained tier and materialize the inside triangles. Public
/// `make_cdt` overloads own type deduction; this operation owns the one
/// arrays-to-arrays implementation shared by every constrained overload.
template <bool ReturnIndexMap, typename Index, typename Coord, typename Int,
          typename PointsPolicy, typename EdgesPolicy, typename... BuildArgs>
auto make_constrained_delaunay(const tf::points<PointsPolicy> &points,
                               const tf::edges<EdgesPolicy> &edges,
                               BuildArgs &&...build_args) {
  tf::constrained_delaunay_triangulator<
      Index, Coord, Int, tf::topology::cdt::parallel_delaunay_execution_policy>
      triangulator;
  triangulator.build(points, edges, static_cast<BuildArgs &&>(build_args)...,
                     tf::cdt_region_mode::nesting);

  auto polygons =
      tf::make_polygons(triangulator.faces(), triangulator.converted_points());
  auto interior = tf::make_mapped_range(
      triangulator.region_labels(), [](auto label) { return label % 2 == 1; });

  if constexpr (ReturnIndexMap) {
    auto reindexed =
        tf::reindexed_by_mask<Index>(polygons, interior, tf::return_index_map);
    auto &point_index_map = std::get<2>(reindexed);
    auto composed =
        tf::compose_index_maps(triangulator.index_map(), point_index_map);
    return std::make_pair(std::move(std::get<0>(reindexed)),
                          std::move(composed));
  } else {
    return tf::reindexed_by_mask<Index>(polygons, interior);
  }
}

} // namespace tf::topology::cdt

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

#include "./build_scene_names.hpp"
#include "./certificate_census.hpp"
#include "./partition_names.hpp"
#include "./phase_clock.hpp"
#include "./pool_records.hpp"
#include "./scene_targets.hpp"

#include "../../../core/algorithm/parallel_for_each.hpp"
#include "../../../core/blocked_buffer.hpp"
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "../../canonical_plane.hpp"
#include "../placement_tables.hpp"

#include <cstddef>
#include <utility>

namespace tf::exact::door::pool {

/// THE PARTITION AS THE PLACEMENT READS IT: per original vertex, the exact
/// plane its own target name committed to, or nothing.
///
/// A FEATURE vertex holds nothing here: its position is the rank-3/2
/// cascade's, and this table never speaks for it. A vertex whose target name
/// joined no pool holds nothing either, which is the untouched case and the
/// common one.
template <typename Int> struct input_pools {
  tf::buffer<int> pool_of_vertex;
  tf::buffer<tf::exact::canonical_plane<Int>> plane;
  election_census election;
  certificate_census certificate;
};

/// The whole pooling pass over the door's own input: name the exact planes
/// the faces stand on, partition those names, and state per vertex which
/// committed plane its placement aims at.
///
/// The order is the authority order: the names are built from the ORIGINAL
/// face data and the partition maps that one name space, so nothing
/// downstream rebuilds a name from a pooled one.
template <typename Index, typename Int, typename RealType>
auto pool_input_planes(const placement_tables<Index, Int, RealType> &tables,
                       const tf::blocked_buffer<Index, 3> &corners,
                       Int tolerance, int resolution, input_pools<Int> &pools)
    -> void {
  phase_clock scene;
  pool_names<Int> names;
  scene_targets targets;
  build_scene_names(tables, corners, tolerance, names, targets);
  pools.election.scene_seconds = scene.since();

  pool_partition<Int> partition;
  partition_names<Int>(names, tolerance, resolution, partition, pools.election,
                       pools.certificate);
  pools.plane = std::move(partition.pool_plane);

  // one disjoint store per vertex, so the vertex is the grain
  pools.pool_of_vertex.allocate(tables.points.size());
  tf::parallel_for_each(
      tf::make_sequence_range(tables.points.size()),
      [&targets, &partition, &pools](std::size_t v) {
        const int source = targets.source[v];
        pools.pool_of_vertex[v] =
            targets.feature[v] || source < 0
                ? -1
                : partition.pool_of_name[std::size_t(source)];
      },
      tf::checked);

#ifdef TF_POOL_CENSUS
  for (const auto pool : pools.pool_of_vertex)
    pools.election.redirected_vertices += pool >= 0 ? 1 : 0;
#endif
}

} // namespace tf::exact::door::pool

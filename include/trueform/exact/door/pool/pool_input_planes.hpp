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
#include "../../../core/buffer.hpp"
#include "../../../core/checked.hpp"
#include "../../../core/views/sequence_range.hpp"
#include "../../canonical_plane.hpp"
#include "../placement_tables.hpp"

#include <cstddef>
#include <utility>

namespace tf::exact::door::pool {

/// THE PARTITION AS THE PLACEMENT READS IT: per flat face, the pool its exact
/// name committed to, or `-1`, and per pool the committed plane. A face that
/// pooled answers to that plane from then on — its corners and every vertex
/// admitted against it land on it — which is how a wall the partition made
/// flat stays flat.
template <typename Int> struct input_pools {
  tf::buffer<int> pool_of_face;
  tf::buffer<tf::exact::canonical_plane<Int>> plane;
  election_census election;
  certificate_census certificate;
};

/// The whole pooling pass over the door's own input: name the exact planes
/// the faces stand on, partition those names, and state per face which
/// committed plane it now answers to.
///
/// The order is the authority order: the names are built from the ORIGINAL
/// face data and the partition maps that one name space, so nothing
/// downstream rebuilds a name from a pooled one.
template <typename Index, typename Int, typename RealType>
auto pool_input_planes(const placement_tables<Index, Int, RealType> &tables,
                       Int tolerance, int resolution, input_pools<Int> &pools)
    -> void {
  phase_clock scene;
  pool_names<Int> names;
  scene_targets targets;
  tf::buffer<int> face_name;
  build_scene_names(tables, tolerance, names, targets, face_name);
  pools.election.scene_seconds = scene.since();

  pool_partition<Int> partition;
  partition_names<Int>(names, tolerance, resolution, partition, pools.election,
                       pools.certificate);
  pools.plane = std::move(partition.pool_plane);

  // one disjoint store per face, so the face is the grain
  pools.pool_of_face.allocate(face_name.size());
  tf::parallel_for_each(
      tf::make_sequence_range(face_name.size()),
      [&face_name, &partition, &pools](std::size_t f) {
        const int name = face_name[f];
        pools.pool_of_face[f] =
            name < 0 ? -1 : partition.pool_of_name[std::size_t(name)];
      },
      tf::checked);

#ifdef TF_POOL_CENSUS
  for (std::size_t v = 0; v < tables.points.size(); ++v)
    for (auto k = tables.incidence_offsets[v];
         k < tables.incidence_offsets[v + 1]; ++k)
      if (pools.pool_of_face[std::size_t(tables.incidence[std::size_t(k)])] >=
          0) {
        ++pools.election.redirected_vertices;
        break;
      }
#endif
}

} // namespace tf::exact::door::pool

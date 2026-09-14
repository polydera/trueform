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

#include "./certificate_census.hpp"
#include "./certificate_lane.hpp"
#include "./find_speaking_cells.hpp"
#include "./group_witnesses_by_plane.hpp"
#include "./intercept_ladder.hpp"
#include "./line_cells.hpp"
#include "./make_intercept_ladder.hpp"
#include "./make_line_cells.hpp"
#include "./phase_clock.hpp"
#include "./pool_records.hpp"
#include "./publish_line_pools.hpp"
#include "./walk_ladder_runs.hpp"

#include "../../../core/algorithm/parallel_fill.hpp"
#include "../../../core/buffer.hpp"

#include <cstdint>

namespace tf::exact::door::pool {

/// THE PARTITION: which original names give up their own plane for which
/// other original face's exact plane.
///
/// ONE AXIS, AND IT IS A LINE. A fixed chart cell is the pool: every name
/// whose direction rounds into it, elected down to one original normal and
/// one original support point, so every member of the cell has a common
/// signed coordinate along one ray. Sorting that coordinate turns the
/// scattered directions into a ladder, and everything above it is one
/// dimensional.
///
/// The phase order is what keeps the certificate rare: a cell is grouped and
/// elected before any certificate exists, and the ladder cuts every pair
/// further apart than the proposal band, so what reaches a solve is a pair
/// already close along one line.
template <typename Int>
auto partition_names(const pool_names<Int> &names, Int tolerance,
                     int resolution, pool_partition<Int> &partition,
                     election_census &census,
                     certificate_census &certificates) -> void {
  const auto count = names.plane.size();
  census.names = std::int64_t(count);
  census.witnesses = std::int64_t(names.witness.size());
  if (count == 0) {
    partition.pool_of_name.clear();
    partition.pool_plane.clear();
    return;
  }

  phase_clock phase;

  tf::buffer<int> witness_offsets;
  group_witnesses_by_plane<Int>(names, witness_offsets);
  census.witness_seconds = phase.since();

  line_cells<Int> cells;
  make_line_cells<Int>(names, resolution, census, cells);

  tf::buffer<char> speaks;
  find_speaking_cells<Int>(cells, witness_offsets, census, speaks);

  // one line per cell, so one frame per cell however many planes are stated
  // on it
  tf::buffer<certificate_lane<Int>> lane;
  lane.allocate(cells.key.size());
  tf::parallel_fill(lane, certificate_lane<Int>{});
  census.cell_seconds = phase.since();

  intercept_ladder<Int> ladder;
  make_intercept_ladder<Int>(names, cells, speaks, tolerance, census, ladder);
  census.ladder_seconds = phase.since();

  tf::buffer<int> owner_of_name;
  walk_ladder_runs<Int>(names, cells, ladder, tolerance, witness_offsets, lane,
                        certificates, census, owner_of_name);
  census.walk_seconds = phase.since();

  publish_line_pools<Int>(names, cells, owner_of_name, census, partition);
  census.publish_seconds = phase.since();
}

} // namespace tf::exact::door::pool

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

#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./plane_piece_key.hpp"
#include "./plane_tier_definitions.hpp"
#include "tbb/parallel_sort.h"

#include <array>
#include <cstddef>
#include <iterator>
#include <utility>

namespace tf::arrangement {

/// Restate promoted faces' tables, one block per face, as the blocks of the
/// planes `plane_of` seats them on (@ref
/// tf::arrangement::seat_plane_promoted_faces): block `k` is plane
/// `planes[k]`'s, ascending. A plane the faces found holds the rows of every
/// face seated on it; a standing plane holds its own rows beside the ones it
/// gained, carried as the world's (@ref
/// tf::arrangement::plane_tier_definitions), so a group the faces never name
/// stays the world's. Every block is in key order.
template <typename Index, typename Int, typename World>
auto seat_plane_entrant_blocks(
    const World &world, const tf::buffer<Index> &plane_of,
    tf::intersect::graph::plane_tables<Index, Int> &entrants,
    tf::buffer<Index> &planes) -> void {
  const auto n_world_planes = world.n_planes();
  tf::buffer<std::array<Index, 2>> seats;
  seats.allocate(plane_of.size());
  tf::parallel_for_each(
      tf::make_sequence_range(plane_of.size()),
      [&](std::size_t face) { seats[face] = {plane_of[face], Index(face)}; },
      tf::checked);
  tbb::parallel_sort(seats.begin(), seats.end());
  tf::buffer<Index> seat_offsets;
  tf::compute_offsets(seats, std::back_inserter(seat_offsets), Index(0),
                      [](const std::array<Index, 2> &x,
                         const std::array<Index, 2> &y) { return x[0] == y[0]; });
  const auto n_blocks =
      seat_offsets.size() == 0 ? std::size_t(0) : seat_offsets.size() - 1;

  planes.allocate(n_blocks);
  tf::buffer<Index> offsets;
  offsets.allocate(n_blocks + 1);
  offsets[0] = Index(0);
  tf::parallel_for_each(
      tf::make_sequence_range(n_blocks),
      [&](std::size_t block) {
        const auto plane = seats[std::size_t(seat_offsets[block])][0];
        planes[block] = plane;
        auto count = plane < n_world_planes
                         ? Index(world.plane_edges(plane).size())
                         : Index(0);
        for (auto at = seat_offsets[block]; at < seat_offsets[block + 1]; ++at)
          count += Index(entrants.plane_edges(seats[std::size_t(at)][1]).size());
        offsets[block + 1] = count;
      },
      tf::checked);
  for (std::size_t block = 1; block < offsets.size(); ++block)
    offsets[block] += offsets[block - 1];

  tf::buffer<Index> rows;
  rows.allocate(std::size_t(offsets[n_blocks]));
  const auto tier =
      make_plane_tier_definitions(world.tables(), entrants, true);
  struct local_t {
    tf::buffer<std::array<Index, 5>> keyed;
  };
  tf::parallel_for_each(
      tf::make_sequence_range(n_blocks),
      [&](std::size_t block, local_t &local) {
        const auto plane = planes[block];
        auto write = std::size_t(offsets[block]);
        if (plane < n_world_planes)
          for (const auto row : world.plane_edges(plane))
            rows[write++] =
                plane_tier_definitions<Index, Int>::carried(Index(row));
        for (auto at = seat_offsets[block]; at < seat_offsets[block + 1]; ++at)
          for (const auto row :
               entrants.plane_edges(seats[std::size_t(at)][1]))
            rows[write++] = row;
        sort_plane_block_by_key(tier, rows, std::size_t(offsets[block]), write,
                                local.keyed);
      },
      local_t{});
  entrants.edges().offsets_buffer() = std::move(offsets);
  entrants.edges().data_buffer() = std::move(rows);
}

} // namespace tf::arrangement

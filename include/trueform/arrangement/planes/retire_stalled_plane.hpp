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

#include "./plane_arrangement_arena.hpp"

#include <array>
#include <cstddef>

namespace tf::arrangement {

/// A carrier the wave has stopped asking holds NO PRODUCT — the state a
/// plane that refused its last round is already in.
///
/// A stalled carrier is not in that state by itself: it can still hold
/// the triangles of a round it produced BEFORE it began refusing, while
/// the local tier block its ticket names has been rebuilt under them
/// since. Those triangles name piece tickets that block no longer
/// states, and the boundary lift reads them as a carrier whose slots
/// cannot be resolved.
///
/// So the retire drops the stale product and nothing else: the plane's
/// own triangle, coplanar and steiner ranges collapse to empty, and so
/// does every member face's range into them. The failed set is NOT
/// touched — @ref tf::arrangement::update_plane_failures is its one
/// producer and has already published this plane every round it refused.
template <typename Index, typename Int, typename World>
auto retire_stalled_plane(const World &world, Index plane,
                          plane_arrangement_arena<Index, Int> &arena) -> void {
  const auto collapse = [](std::array<Index, 2> &range) {
    range[1] = range[0];
  };
  collapse(arena.range[std::size_t(plane)]);
  collapse(arena.cop_range[std::size_t(plane)]);
  collapse(arena.stn_range[std::size_t(plane)]);
  const auto members = world.member_count(plane);
  for (Index member = 0; member < members; ++member)
    collapse(arena.face_range[std::size_t(world.member(plane, member))]);
}

} // namespace tf::arrangement

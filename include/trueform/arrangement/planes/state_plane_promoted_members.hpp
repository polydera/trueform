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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::arrangement {

/// Per plane of the whole carrier space, the promoted faces seated on it, in
/// face order: `offsets` over `n_planes`, `members` the faces themselves.
/// `plane_of[i]` is promoted face `n_base_faces + i`'s plane.
template <typename Index>
auto state_plane_promoted_members(const tf::buffer<Index> &plane_of,
                                  Index n_base_faces, Index n_planes,
                                  tf::buffer<Index> &offsets,
                                  tf::buffer<Index> &members) -> void {
  const auto n = plane_of.size();
  tf::buffer<std::array<Index, 2>> seats;
  seats.allocate(n);
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&](std::size_t i) {
        seats[i] = {plane_of[i], n_base_faces + Index(i)};
      },
      tf::checked);
  tbb::parallel_sort(seats.begin(), seats.end());
  offsets.allocate(std::size_t(n_planes) + 1);
  tf::parallel_fill(offsets, Index(0));
  members.allocate(n);
  tf::parallel_for_each(
      tf::make_sequence_range(n),
      [&](std::size_t at) {
        members[at] = seats[at][1];
        if (at + 1 == n || seats[at + 1][0] != seats[at][0])
          offsets[std::size_t(seats[at][0]) + 1] = Index(at + 1);
      },
      tf::checked);
  // a plane no promoted face seats on ends where the plane before it ends
  for (std::size_t p = 1; p < offsets.size(); ++p)
    offsets[p] = std::max(offsets[p], offsets[p - 1]);
}

} // namespace tf::arrangement

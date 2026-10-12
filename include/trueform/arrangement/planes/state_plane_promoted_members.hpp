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

#include "../../core/buffer.hpp"

#include <algorithm>
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
  offsets.allocate(std::size_t(n_planes) + 1);
  std::fill(offsets.begin(), offsets.end(), Index(0));
  for (const auto plane : plane_of)
    ++offsets[std::size_t(plane) + 1];
  for (std::size_t p = 1; p < offsets.size(); ++p)
    offsets[p] += offsets[p - 1];
  members.allocate(plane_of.size());
  tf::buffer<Index> cursor;
  cursor.allocate(std::size_t(n_planes));
  std::copy(offsets.begin(), offsets.end() - 1, cursor.begin());
  for (std::size_t i = 0; i < plane_of.size(); ++i)
    members[std::size_t(cursor[std::size_t(plane_of[i])]++)] =
        n_base_faces + Index(i);
}

} // namespace tf::arrangement

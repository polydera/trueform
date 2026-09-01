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

#include <cstddef>

namespace tf::arrangement {

/// CORE. Resolve one immutable canonical group through the router: `-1` while
/// the immutable group is still the authority, `-2` when a wave RETIRED the
/// root, and its current local group otherwise.
///
/// `absent` is what a router that does not state the group answers — none
/// built yet, or an id past the extent this router covers.
template <typename Index>
auto resolve_plane_group_router(const tf::buffer<Index> &router,
                                Index immutable_group, Index absent) -> Index {
  return immutable_group < Index(0) ||
                 std::size_t(immutable_group) >= router.size()
             ? absent
             : router[std::size_t(immutable_group)];
}

} // namespace tf::arrangement

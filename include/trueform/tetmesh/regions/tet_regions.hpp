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

namespace tf::tetmesh {

/// The winding number of every finite cell of a recovered complex: how many
/// times the constraint surface wraps it, counted with orientation, the
/// unbounded region reading zero. `consistent` is false when two crossings
/// into one region disagree, which an open or inconsistently wound surface
/// states; the numbers are then not stated.
template <typename Index> struct tet_regions {
  tf::buffer<int> winding;
  tf::buffer<Index> regions;
  bool consistent = false;
};

} // namespace tf::tetmesh

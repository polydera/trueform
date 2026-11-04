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
#include "../../../exact/vertex.hpp"
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// A weld record: the position a build takes, which stands on the frame and
/// so is its floor (@ref tet_shallow_position), canonical name, original
/// input slot and the Morton key the weld orders it by
/// (@ref tet_site_morton_key).
template <typename Index, typename Coord> struct tet_site {
  tf::exact::pt3<Coord> floor;
  Index id;
  Index input;
  std::uint64_t key = 0;
};

} // namespace tf::topology::cdt::dt3

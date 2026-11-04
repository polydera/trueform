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
#include "../tet_position.hpp"
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// A split one round asks of a subsegment row: its position, minted for a
/// fresh one, its numerator on the segment's frame, `-1` off it, the site it
/// lands on, `-1` for a fresh one, and the ring it realizes, `2 * segment +
/// end` of the shelled corner's end, `-1` for none.
template <typename Index, typename Coord> struct tet_subsegment_split {
  Index subsegment;
  std::int64_t numerator;
  tet_position<Coord> position;
  Index site;
  Index ring = -1;
};

} // namespace tf::topology::cdt::dt3

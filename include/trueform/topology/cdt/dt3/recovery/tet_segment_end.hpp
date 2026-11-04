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
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// One end of a protected segment: the site it stands on, the segment, and
/// which end, sorted by site so a corner's segments are one run.
template <typename Index> struct tet_segment_end {
  Index site;
  Index segment;
  std::uint8_t end;
};

} // namespace tf::topology::cdt::dt3

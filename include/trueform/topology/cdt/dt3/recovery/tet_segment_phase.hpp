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
#include "../../../../core/buffer.hpp"
#include "../../../../core/offset_block_buffer.hpp"
#include "./tet_segment_recovery.hpp"
#include <array>

namespace tf::topology::cdt::dt3 {

/// Phase S's product and the scratch its passes reuse: `segments`, the
/// protected segments and every edge bounding a face, each once with its
/// slots ascending; `edges`, a block per face, the segments its rim runs
/// along; `holders`, aligned with `edges` entry for entry, the face's
/// triangle holding that edge, as the edge it winds and its third corner;
/// and `segment`, the recovery whose chains, statuses and rounds state what
/// became of each.
template <typename Index, typename Coord> struct tet_segment_phase {
  tf::buffer<std::array<Index, 2>> segments;
  tf::offset_block_buffer<Index, Index> edges;
  tf::buffer<std::array<Index, 3>> holders;
  tet_segment_recovery<Index, Coord> segment;
  tf::buffer<std::array<Index, 2>> pieces;
};

} // namespace tf::topology::cdt::dt3

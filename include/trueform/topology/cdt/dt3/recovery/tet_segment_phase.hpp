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
#include "./tet_facet_segment_scratch.hpp"
#include "./regions/tet_patch_tables.hpp"
#include "./tet_segment_recovery.hpp"
#include "./tet_site_segments.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Phase S's product and the scratch its passes reuse: `segments`, the
/// protected segments and every edge bounding a face, each once with its
/// slots ascending, and `segment_sites` them per lower site
/// (@ref tet_site_segments); `edges`, a block per face, the segments its rim
/// runs along; `holders`, aligned with `edges` entry for entry, the face's
/// triangle holding that edge, as the edge it winds and its third corner;
/// `segment`, the recovery whose chains, statuses and rounds state what
/// became of each; `facet_segments`, the tables the statement of the
/// three above walks on (@ref state_tet_facet_segments); and `rim_faces`,
/// per segment the live faces whose rim runs along it, counted over the
/// faces below `rim_counted` (@ref retire_tet_segments); they live the
/// build, each retirement a diff on them.
template <typename Index, typename Coord> struct tet_segment_phase {
  tf::buffer<std::array<Index, 2>> segments;
  tet_site_segments<Index> segment_sites;
  tf::offset_block_buffer<Index, Index> edges;
  tf::buffer<Index> edge_block;
  tf::buffer<std::array<Index, 3>> holders;
  tet_patch_tables<Index> region_tables;
  tet_segment_recovery<Index, Coord> segment;
  tet_facet_segment_scratch<Index> facet_segments;
  tf::buffer<Index> rim_faces;
  std::size_t rim_counted = 0;
};

} // namespace tf::topology::cdt::dt3

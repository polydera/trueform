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
#include "./tet_reach_cell_key.hpp"
#include "../../core/buffer.hpp"
#include "../../core/points_buffer.hpp"
#include <array>

namespace tf::tetmesh {

/// The tables the entrance's workers walk on, constructed once by the entry
/// (@ref make_tet_entrance_surface) and handed down: the merged triangles
/// with their origins, facets and slots in facet order and the facets' fill
/// cursor, and the reach cells' keys, survivors, dense numbering and kept
/// points (@ref state_tet_reach_cells). The merge's own tables
/// (@ref tet_facet_workspace) are not here: the entry releases them before
/// the surface's are built.
template <typename Index, typename Int> struct tet_entrance_workspace {
  tf::buffer<std::array<Index, 3>> triangles;
  tf::buffer<Index> origin;
  tf::buffer<Index> facet_of;
  tf::buffer<Index> slot;
  tf::buffer<Index> cursor;
  tf::buffer<tet_reach_cell_key<Index, Int>> cell_keys;
  tf::buffer<Index> cell_point;
  tf::buffer<Index> cell_dense;
  tf::points_buffer<Int, 3> cell_points;
};

} // namespace tf::tetmesh

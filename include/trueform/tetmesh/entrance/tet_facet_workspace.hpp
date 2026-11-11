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
#include "./count_tet_open_edges.hpp"

namespace tf::tetmesh {

/// Two triangles the facet merge joins into one facet, the lower first.
template <typename Index> struct tet_facet_join {
  Index low;
  Index high;
};

/// The tables the facet merge (@ref merge_tet_facets) walks on, constructed
/// by its owner and handed in: per triangle the way its normal faces
/// and its kind; the joins, the classes they close and the facets'
/// renumbering; and the edge steps the merged triangles' open edges are
/// counted on (@ref count_tet_merged_open_edges).
template <typename Index, typename Int> struct tet_facet_workspace {
  tf::buffer<signed char> facing;
  tf::buffer<signed char> kind;
  tf::buffer<tet_facet_join<Index>> joins;
  tf::buffer<Index> classes;
  tf::buffer<Index> renumber;
  tf::buffer<tet_edge_step<Index>> steps;
};

} // namespace tf::tetmesh

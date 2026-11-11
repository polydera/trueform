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
#include "../../../../../core/buffer.hpp"
#include "../../tet_claim_words.hpp"
#include <array>

namespace tf::topology::cdt::dt3 {

/// The tables a partition of the surface walks on
/// (@ref partition_tet_surface, @ref claim_tet_patch_ranges,
/// @ref join_tet_patch_pockets, @ref make_tet_first_facet_patches,
/// @ref make_tet_patch_adjacency): `order` the named triangles in facet
/// order, `keys` a sample of them by smallest corner slot, `bounds` the
/// ranges' ends among those keys and `seeds` the patches' first triangles;
/// `range` every triangle's range; `border`, `queue` and `hops` the pockets'
/// join; `claims` the smallest facet naming every id; and `pairs` the
/// patches meeting across an edge.
template <typename Index> struct tet_patch_workspace {
  tf::buffer<Index> order;
  tf::buffer<std::array<Index, 2>> keys;
  tf::buffer<std::array<Index, 2>> bounds;
  tf::buffer<Index> seeds;
  tf::buffer<Index> range;
  tf::buffer<std::array<Index, 2>> border;
  tf::buffer<Index> queue;
  tf::buffer<Index> hops;
  tet_claim_words claims;
  tf::buffer<std::array<Index, 2>> pairs;
};

} // namespace tf::topology::cdt::dt3

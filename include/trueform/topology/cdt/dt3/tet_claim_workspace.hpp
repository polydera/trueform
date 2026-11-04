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
#include "../../../core/buffer.hpp"
#include "./tet_cavity_links.hpp"
#include "./tet_claim_stream.hpp"
#include "./tet_claim_words.hpp"
#include <array>
#include <vector>

namespace tf::topology::cdt::dt3 {

/// Spatial streams mutate preallocated rows under temporary cell ownership.
/// Every mutable read holds the cell; a completed cavity also holds its outside
/// neighbors and coplanar hull predicate dependencies. Failed attempts release
/// all claims before retrying. Only the barrier transfers free rows or
/// publishes the state lane.
template <typename Index> struct tet_claim_workspace {
  tet_claim_words ownership;
  tf::buffer<Index> free_next, tail;
  tf::buffer<tet_claim_stream<Index>> streams;
  /// Per stream, the edges its cavities took while the owner records them,
  /// merged into the owner at the barrier.
  std::vector<tf::buffer<std::array<Index, 2>>> killed;
  /// Per stream, the pairing of its cavities' boundaries.
  std::vector<tet_cavity_links<Index>> links;
};

} // namespace tf::topology::cdt::dt3

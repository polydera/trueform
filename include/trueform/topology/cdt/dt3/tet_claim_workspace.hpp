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
#include "../../../core/cache_aligned_slot.hpp"
#include "../../../core/memory.hpp"
#include "./tet_cavity_links.hpp"
#include "./tet_claim_scratch.hpp"
#include "./tet_claim_stream.hpp"
#include "./tet_claim_words.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Spatial streams mutate pooled rows under temporary cell ownership.
/// Every mutable read holds the cell; a completed cavity also holds its outside
/// neighbors and coplanar hull predicate dependencies. A failed attempt
/// releases all its claims. A stream holds the rows of its pool, each dead, so
/// no other stream reaches them; only a barrier moves or grows them.
template <typename Index> struct tet_claim_workspace {
  tet_claim_words ownership;
  tf::buffer<Index> free_next, tail;
  tf::buffer<tet_claim_stream<Index>> streams;
  /// The sites the complex holds as of the last band; with its live rows it
  /// is the density the pools are grown by.
  std::size_t inserted = 0;
  /// Per stream, the edges its cavities took while the owner records them,
  /// merged into the owner at the barrier.
  tf::core::std_vector<tf::buffer<std::array<Index, 2>>> killed;
  /// Per stream, the pairing of its cavities' boundaries.
  tf::core::std_vector<tet_cavity_links<Index>> links;
  /// Per stream, the lists one attempt walks on, cleared at the attempt and
  /// kept across the rounds of every band. A list writes its end on every
  /// push, so the rows stand a cache line apart.
  tf::core::std_vector<tf::core::cache_aligned_slot<tet_claim_scratch<Index>>>
      scratch;
};

} // namespace tf::topology::cdt::dt3

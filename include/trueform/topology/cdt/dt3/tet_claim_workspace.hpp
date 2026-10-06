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
#include "./tet_claim_stream.hpp"
#include "./tet_claim_words.hpp"

namespace tf::topology::cdt::dt3 {

/// Spatial streams mutate preallocated rows under temporary cell ownership.
/// Every mutable read holds the cell; a completed cavity also holds its outside
/// neighbors and coplanar hull predicate dependencies. Failed attempts release
/// all claims before retrying. Only the barrier transfers free rows or
/// publishes the state lane.
///
/// Sustained insertion follows Geogram PDEL, revision
/// 69803f75c7e3cc3b78fea2906b9b0a15d3ec8401:
/// https://github.com/BrunoLevy/geogram/blob/69803f75c7e3cc3b78fea2906b9b0a15d3ec8401/src/lib/geogram/delaunay/parallel_delaunay_3d.cpp
template <typename Index> struct tet_claim_workspace {
  tet_claim_words ownership;
  tf::buffer<Index> free_next, tail;
  tf::buffer<tet_claim_stream<Index>> streams;
};

} // namespace tf::topology::cdt::dt3

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
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// The segments of a table's prefix `[0, chained)` per lower site, in the
/// order they were stated: `head[a]` the first segment whose lower site is
/// `a`, `tail[a]` the last, `next[s]` the one stated after `s` with the same
/// lower site, `-1` ending a chain. A statement's run is chained by the next
/// statement, the one that looks it up (@ref state_tet_facet_segments). The
/// chains live the build and only grow: a retired segment stays chained,
/// and the lookup asks whether a segment still names its edge.
template <typename Index> struct tet_site_segments {
  tf::buffer<Index> head;
  tf::buffer<Index> tail;
  tf::buffer<Index> next;
  std::size_t chained = 0;
};

} // namespace tf::topology::cdt::dt3

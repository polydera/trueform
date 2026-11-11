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
#include <array>

namespace tf::topology::cdt::dt3 {

/// One edge of one triangle of a face, named by its site slots ascending,
/// `at` the slot `3 t + k` it was stated from, or `-1` for a segment the
/// caller gave (@ref state_tet_facet_segments).
template <typename Index> struct tet_facet_segment_record {
  std::array<Index, 2> key;
  Index at;
};

/// The tables one statement of a facet's segments walks on: `facet_of` the
/// face of every triangle and `first` each face's first, `records` the
/// edges with `record_scratch` and `runs` the bucketing's own
/// (@ref bucket_tet_records), `counts` its blocked counts, `segment_at` the
/// segment every stated edge names among the statement's own `keys`, and
/// `global_of` each of those keys' id in the table
/// (@ref state_tet_facet_segments).
template <typename Index> struct tet_facet_segment_scratch {
  tf::buffer<Index> facet_of;
  tf::buffer<Index> first;
  tf::buffer<tet_facet_segment_record<Index>> records;
  tf::buffer<tet_facet_segment_record<Index>> record_scratch;
  tf::buffer<Index> runs;
  tf::buffer<Index> counts;
  tf::buffer<Index> segment_at;
  tf::buffer<std::array<Index, 2>> keys;
  tf::buffer<Index> global_of;
};

} // namespace tf::topology::cdt::dt3

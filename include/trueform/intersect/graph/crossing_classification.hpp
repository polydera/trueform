/*
 * Copyright (c) 2025 XLAB
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
#include "../../core/range.hpp"
#include "./crossing_record.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>
#include <tuple>

namespace tf::intersect::graph {

/// Sort crossing records by (type, type-specific key).
/// EE sorted by (triple, edge pair) — the pair tiebreak keeps coplanar-pack
/// subgroups contiguous and makes the group representative deterministic.
/// VE by (edge, point), VV by (point_a, point_b).
template <typename Index>
auto sort_crossing_records(tf::buffer<crossing_record<Index>> &records)
    -> void {
  tbb::parallel_sort(
      records.begin(), records.end(),
      [](const crossing_record<Index> &a, const crossing_record<Index> &b) {
        if (a.type != b.type)
          return a.type < b.type;
        if (a.type == crossing_record<Index>::ee) {
          if (a.triple != b.triple)
            return a.triple < b.triple;
          return std::tie(a.edge_a, a.edge_b) < std::tie(b.edge_a, b.edge_b);
        }
        if (a.type == crossing_record<Index>::ve)
          return std::tie(a.edge_a, a.point_a) < std::tie(b.edge_a, b.point_a);
        return std::tie(a.point_a, a.point_b) < std::tie(b.point_a, b.point_b);
      });
}

/// Find type boundaries in sorted crossing records.
template <typename Index>
auto find_type_ranges(tf::buffer<crossing_record<Index>> &records) {
  auto ve_start = std::lower_bound(
      records.begin(), records.end(), crossing_record<Index>::ve,
      [](const crossing_record<Index> &r,
         typename crossing_record<Index>::type_t t) { return r.type < t; });
  auto vv_start = std::lower_bound(
      ve_start, records.end(), crossing_record<Index>::vv,
      [](const crossing_record<Index> &r,
         typename crossing_record<Index>::type_t t) { return r.type < t; });
  return std::make_tuple(tf::make_range(records.begin(), ve_start),
                         tf::make_range(ve_start, vv_start),
                         tf::make_range(vv_start, records.end()));
}

/// Deduplicate VE records (same edge + same point). Returns shortened range.
template <typename VERange>
auto dedup_ve(VERange &&ve_range) {
  auto new_end =
      std::unique(ve_range.begin(), ve_range.end(),
                  [](const auto &a, const auto &b) {
                    return a.edge_a == b.edge_a && a.point_a == b.point_a;
                  });
  return tf::make_range(ve_range.begin(), new_end);
}

} // namespace tf::intersect::graph

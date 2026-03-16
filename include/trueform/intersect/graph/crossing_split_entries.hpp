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

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "./crossing_record.hpp"

namespace tf::intersect::graph {

/// Collect split entries from EE + VE crossing records (parallel).
///
/// EE records produce 2 entries each (one per edge), with point_id =
/// crossing_base + group index. VE records produce 1 entry each.
/// Output is pre-allocated; EE and VE write to non-overlapping regions.
template <typename Index, typename EERange, typename VERange>
auto collect_split_entries(const EERange &ee_range,
                           const tf::buffer<Index> &ee_offsets,
                           const VERange &ve_range, Index crossing_base)
    -> tf::buffer<edge_split_entry<Index>> {
  auto ee_count = ee_range.size();
  auto ve_count = ve_range.size();
  auto num_ee_groups = ee_offsets.size() > 0 ? ee_offsets.size() - 1 : 0;

  tf::buffer<edge_split_entry<Index>> entries;
  entries.allocate(ee_count * 2 + ve_count);

  // EE: parallel over groups
  if (num_ee_groups > 0) {
    auto ee_groups = tf::make_offset_block_range(ee_offsets, ee_range);
    tf::parallel_for_each(tf::enumerate(ee_groups), [&](auto pair) {
      auto &&[g, group] = pair;
      Index pid = crossing_base + static_cast<Index>(g);
      auto base = ee_offsets[g];
      for (std::size_t i = 0; i < group.size(); ++i) {
        auto &r = group[i];
        entries[2 * (base + i)] = {r.edge_a, pid};
        entries[2 * (base + i) + 1] = {r.edge_b, pid};
      }
    });
  }

  // VE: parallel over records
  auto ve_base = ee_count * 2;
  tf::parallel_for_each(tf::enumerate(ve_range), [&](auto pair) {
    auto &&[i, r] = pair;
    entries[ve_base + i] = {r.edge_a, r.point_a};
  });

  return entries;
}

} // namespace tf::intersect::graph

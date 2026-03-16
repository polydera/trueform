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

#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/views/indirect_range.hpp"
#include "./edge.hpp"
#include "tbb/parallel_sort.h"

namespace tf::intersect::graph {

/// Sort edge instances by canonical key, assign canonical group IDs,
/// and rebuild group offsets.
///
/// Per-face edge lists (`edges`) store instance indices into the flat
/// data buffer. These are remapped to stay valid after sorting.
template <typename Index>
auto canonicalize_edges(
    tf::offset_block_buffer<Index, edge<Index>> &edge_defs,
    tf::offset_block_buffer<Index, Index> &edges) -> void {
  auto &data = edge_defs.data_buffer();
  if (data.size() == 0)
    return;

  tf::parallel_for_each(tf::enumerate(data), [](auto pair) {
    auto &&[i, e] = pair;
    e.id = static_cast<Index>(i);
  });

  auto ck = [](const edge<Index> &e) {
    return std::make_pair(std::min(e.point_0, e.point_1),
                          std::max(e.point_0, e.point_1));
  };

  tbb::parallel_sort(
      data.begin(), data.end(),
      [&](const auto &a, const auto &b) { return ck(a) < ck(b); });

  tf::buffer<Index> map;
  map.allocate(data.size());
  Index new_id = 0;

  auto map_f = [&](auto id, auto new_id) {
    map[data[id].id] = id;
    data[id].id = new_id;
  };
  map_f(0, 0);
  for (std::size_t i = 1; i < data.size(); ++i) {
    if (ck(data[i]) != ck(data[i - 1]))
      ++new_id;
    map_f(i, new_id);
  }

  auto &offsets = edge_defs.offsets_buffer();
  offsets.allocate(new_id + 2);
  tf::compute_offsets(
      data, offsets.begin(), Index(0),
      [](const auto &x, const auto &y) { return x.id == y.id; });

  tf::parallel_copy(tf::make_indirect_range(edges.data_buffer(), map),
                    edges.data_buffer());
}

} // namespace tf::intersect::graph

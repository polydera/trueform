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

#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/offset_block_buffer.hpp"
#include "../../core/reallocate.hpp"
#include "../../topology/topo_type.hpp"
#include "./vertex.hpp"

#include <algorithm>

namespace tf::intersect::graph {

/// Build a "dirty" base loop by inserting `base_edges`' intermediates into
/// `base_loop`. Each entry in `base_edges` is a boundary sub-edge with
/// `ordinal` (position of start vertex in base_loop) and `sub_ordinal`
/// (parametric order along that boundary segment). Entries are sorted in
/// place by `(ordinal, sub_ordinal)`. For each base loop position p, the
/// vertex is appended to `dirty`; then for each boundary entry whose
/// `ordinal == p`, its `point_1` is appended as a created vertex.
///
/// `dirty` is APPENDED to (caller manages its size).
template <typename Index, typename Range, typename Loop>
auto build_dirty_loop(const Range &base_edges, const Loop &base_loop,
                       tf::buffer<vertex<Index>> &dirty) -> void {
  std::sort(base_edges.begin(), base_edges.end(),
            [](const auto &a, const auto &b) {
              return std::make_pair(a.ordinal, a.sub_ordinal) <
                     std::make_pair(b.ordinal, b.sub_ordinal);
            });

  auto bit = base_edges.begin();
  auto bend = base_edges.end();
  for (std::size_t p = 0; p < base_loop.size(); ++p) {
    dirty.push_back(base_loop[p]);
    while (bit != bend && static_cast<std::size_t>(bit->ordinal) == p) {
      auto next = bit + 1;
      bool is_last = (next == bend ||
                      static_cast<std::size_t>(next->ordinal) != p);
      // Skip last sub-edge in the group: its point_1 is loop[p+1],
      // already provided by the outer walk. Otherwise its point_1 is
      // an intermediate split point on this base-loop edge.
      if (!is_last)
        dirty.push_back({vertex_source::created, bit->point_1,
                         {base_loop[p].sub_id.id, tf::topo_type::edge}});
      ++bit;
    }
  }
}

/// Clean a raw base loop into result:
/// - remove consecutive duplicates
/// - snip antennas: (X, A, X) → (X)
/// Handles wrap-around. Appends cleaned vertices to result.
template <typename Index, typename Range>
auto clean_loop(const Range &dirty, tf::buffer<vertex<Index>> &result) -> void {
  using vertex_t = vertex<Index>;
  auto eq = [](const vertex_t &a, const vertex_t &b) {
    return a.id == b.id && a.source == b.source;
  };

  auto start = result.size();
  if (dirty.size() < 3) {
    for (auto &v : dirty)
      result.push_back(v);
    return;
  }

  // Linear pass: dup + antenna (X, A, X) snipping
  for (const auto &v : dirty) {
    // consecutive duplicate
    if (result.size() > start && eq(result.back(), v))
      continue;
    // antenna: just-pushed equals two back → pop A, drop incoming X
    if (result.size() - start >= 2 && eq(result[result.size() - 2], v)) {
      result.pop_back();
      continue;
    }
    result.push_back(v);
  }

  // Wrap-around cleanup
  auto sz = [&] { return result.size() - start; };
  bool changed = true;
  while (changed && sz() >= 2) {
    changed = false;
    // (back, front) duplicate → drop back
    if (eq(result.back(), result[start])) {
      result.pop_back();
      changed = true;
    } else if (sz() >= 3) {
      // (back, front, front+1) = (X, A, X) → drop front A and back X
      if (eq(result.back(), result[start + 1])) {
        // erase front (A)
        for (std::size_t j = start; j + 1 < result.size(); ++j)
          result[j] = result[j + 1];
        result.pop_back(); // shifted: drop one
        result.pop_back(); // drop back X
        changed = true;
      }
      // (back-2, back-1, front) = (X, A, X) → drop back A and back X
      else if (eq(result[result.size() - 2], result[start])) {
        result.pop_back();
        result.pop_back();
        changed = true;
      }
    }
  }
}

/// Clean every loop in `loops` in parallel via `clean_loop`. Rebuilds
/// `loops` data and offsets in place.
template <typename Index>
auto clean_loops(tf::offset_block_buffer<Index, vertex<Index>> &loops)
    -> void {
  auto old_range = tf::make_range(loops);
  tf::buffer<vertex<Index>> new_data;
  tf::buffer<Index> new_offsets;
  new_offsets.allocate(loops.size() + 1);
  new_offsets[0] = 0;
  std::size_t loop_i = 1;

  struct local_t {
    tf::buffer<vertex<Index>> data;
    tf::buffer<Index> sizes;
  };

  auto task = [&](auto &&range, local_t &local) {
    local.sizes.allocate(range.size());
    auto sit = local.sizes.begin();
    for (const auto &loop : range) {
      auto old_size = local.data.size();
      clean_loop<Index>(loop, local.data);
      *sit++ = static_cast<Index>(local.data.size() - old_size);
    }
  };

  auto agg = [&](const local_t &local, const tf::none_t &) {
    tf::core::append(local.data, new_data);
    for (auto sz : local.sizes) {
      new_offsets[loop_i] = new_offsets[loop_i - 1] + sz;
      ++loop_i;
    }
  };

  tf::blocked_reduce_sequenced_aggregate(old_range, tf::none, local_t{},
                                         task, agg);

  loops.data_buffer() = std::move(new_data);
  loops.offsets_buffer() = std::move(new_offsets);
}

} // namespace tf::intersect::graph

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
#include "../../core/buffer.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::linalg {

/// How many blocks per worker the factor's tree is cut into: more balance the
/// blocks, and each one moves work up into the frontier, whose fronts go
/// tile by tile rather than subtree by subtree.
inline constexpr std::size_t cholesky_blocks_per_worker = 1;

/// Cut a postordered elimination tree into subtrees that factor apart and the
/// frontier above them.
///
/// `work` holds each node's own work and receives its subtree's. Starting from
/// the roots, the heaviest subtree gives way to its children until there are
/// `target` subtrees or none is heavier than the whole work over `target`;
/// the nodes that gave way are the frontier. `schedule` lists each kept
/// subtree's nodes, a contiguous run, in ascending order and then the
/// frontier ascending; `block_offsets` partitions its subtrees, and its last
/// offset is where the frontier starts.
template <typename Index>
auto partition_elimination_tree(const tf::buffer<Index> &parent,
                                std::size_t target, tf::buffer<double> &work,
                                tf::buffer<Index> &size,
                                tf::buffer<Index> &heap,
                                tf::buffer<Index> &schedule,
                                tf::buffer<Index> &block_offsets) -> void {
  const std::size_t n = parent.size();
  size.allocate_and_initialize(n, Index(1));
  heap.clear();
  double total = 0.0;
  for (std::size_t j = 0; j < n; ++j) {
    const Index up = parent[j];
    if (up == Index(-1)) {
      heap.push_back(Index(j));
      total += work[j];
      continue;
    }
    work[std::size_t(up)] += work[j];
    size[std::size_t(up)] = Index(size[std::size_t(up)] + size[j]);
  }

  auto lighter = [&](Index a, Index b) {
    return work[std::size_t(a)] < work[std::size_t(b)] ||
           (work[std::size_t(a)] == work[std::size_t(b)] && a < b);
  };
  std::make_heap(heap.begin(), heap.end(), lighter);
  const double share = total / double(std::max<std::size_t>(target, 1));
  while (heap.size() && heap.size() < target) {
    const Index top = heap.front();
    if (size[std::size_t(top)] == Index(1) || work[std::size_t(top)] <= share)
      break;
    std::pop_heap(heap.begin(), heap.end(), lighter);
    heap.pop_back();
    const Index first = Index(top - size[std::size_t(top)] + 1);
    for (Index child = Index(top - 1); child >= first;
         child = Index(child - size[std::size_t(child)])) {
      heap.push_back(child);
      std::push_heap(heap.begin(), heap.end(), lighter);
    }
    // A frontier node keeps no run of its own.
    size[std::size_t(top)] = Index(0);
  }

  std::sort(heap.begin(), heap.end());
  schedule.allocate(n);
  block_offsets.allocate(heap.size() + 1);
  std::size_t at = 0;
  for (std::size_t b = 0; b < heap.size(); ++b) {
    block_offsets[b] = Index(at);
    const std::size_t root = std::size_t(heap[b]);
    for (std::size_t j = root + 1 - std::size_t(size[root]); j <= root; ++j)
      schedule[at++] = Index(j);
  }
  block_offsets[heap.size()] = Index(at);
  for (std::size_t j = 0; j < n; ++j)
    if (size[j] == Index(0))
      schedule[at++] = Index(j);
}

} // namespace tf::linalg

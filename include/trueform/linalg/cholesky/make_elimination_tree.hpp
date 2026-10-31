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
#include <cstddef>

namespace tf::linalg {

/// The elimination tree of a symmetric graph eliminated in `order`, by
/// position: `parent[k]` is the position whose row first takes a fill entry
/// from position `k`, or -1 at a root.
///
/// Each position climbs from every earlier neighbour to the root of its
/// current subtree, compressing the path onto itself through `ancestor`.
template <typename Index>
auto make_elimination_tree(const tf::buffer<Index> &offsets,
                           const tf::buffer<Index> &adjacency,
                           const tf::buffer<Index> &order,
                           const tf::buffer<Index> &rank,
                           tf::buffer<Index> &ancestor,
                           tf::buffer<Index> &parent) -> void {
  const std::size_t n = order.size();
  ancestor.allocate(n);
  parent.allocate(n);
  for (std::size_t k = 0; k < n; ++k) {
    parent[k] = Index(-1);
    ancestor[k] = Index(-1);
    const std::size_t row = std::size_t(order[k]);
    for (Index at = offsets[row]; at < offsets[row + 1]; ++at) {
      Index i = rank[std::size_t(adjacency[std::size_t(at)])];
      while (i != Index(-1) && std::size_t(i) < k) {
        const Index up = ancestor[std::size_t(i)];
        ancestor[std::size_t(i)] = Index(k);
        if (up == Index(-1))
          parent[std::size_t(i)] = Index(k);
        i = up;
      }
    }
  }
}

} // namespace tf::linalg

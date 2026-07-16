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

namespace tf::csg::graph {

/// @ingroup csg
/// @brief Union-find over dense ids with path halving and union by
///        minimum root: the root of every set is its smallest element,
///        so `find` is canonical regardless of merge order.
template <typename Index> struct union_find {
  tf::buffer<Index> parent;

  auto reset(std::size_t n) {
    parent.allocate(n);
    for (std::size_t i = 0; i < n; ++i)
      parent[i] = Index(i);
  }

  auto find(Index x) -> Index {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  auto unite(Index a, Index b) {
    Index ra = find(a), rb = find(b);
    if (ra != rb)
      parent[std::size_t(std::max(ra, rb))] = std::min(ra, rb);
  }
};

} // namespace tf::csg::graph

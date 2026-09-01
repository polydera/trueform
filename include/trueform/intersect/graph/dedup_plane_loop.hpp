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
#include "./vertex.hpp"

namespace tf::intersect::graph {

/// A raw base loop with its consecutive and wrap-around duplicates
/// removed. Antennas stay in: a spike is one wall walked BOTH WAYS, so its
/// two sides are AB and BA — two instances of one canonical group, which
/// the arrangement carries whole, their multiplicity being the wall's
/// parity.
template <typename Index, typename Range>
auto dedup_plane_loop(const Range &dirty, tf::buffer<vertex<Index>> &result)
    -> void {
  using vertex_t = vertex<Index>;
  auto eq = [](const vertex_t &a, const vertex_t &b) {
    return a.id == b.id && a.source == b.source;
  };
  const auto start = result.size();
  if (dirty.size() < 3) {
    for (const auto &v : dirty)
      result.push_back(v);
    return;
  }
  for (const auto &v : dirty)
    if (!(result.size() > start && eq(result.back(), v)))
      result.push_back(v);
  while (result.size() - start >= 2 && eq(result.back(), result[start]))
    result.pop_back();
}

} // namespace tf::intersect::graph

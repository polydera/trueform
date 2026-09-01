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
#include "./delaunay_execution_tuning.hpp"
#include "./partition_morton_sites.hpp"
#include <cstddef>

namespace tf::topology::cdt {

template <typename KeyBuffer, typename Visitor>
auto visit_morton_partitions(const KeyBuffer &keys, std::size_t first,
                             std::size_t last, Visitor &visitor) -> void {
  if (last - first <= delaunay_execution_tuning::leaf_sites) {
    visitor(first, last);
    return;
  }

  const auto partition = partition_morton_sites(keys, first, last);
  if (!partition) {
    visitor(first, last);
    return;
  }

  visit_morton_partitions(keys, first, partition->cut, visitor);
  visit_morton_partitions(keys, partition->cut, last, visitor);
}

template <typename KeyBuffer, typename Visitor>
auto visit_morton_partitions(const KeyBuffer &keys, Visitor &&visitor) -> void {
  if (keys.size() == 0)
    return;
  auto &&operation = visitor;
  visit_morton_partitions(keys, 0, keys.size(), operation);
}

} // namespace tf::topology::cdt

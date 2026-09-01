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
#include "./visit_morton_partitions.hpp"
#include <algorithm>
#include <cstddef>

namespace tf::topology::cdt {

template <typename Points, typename KeyBuffer>
auto sort_parallel_fallback_leaves(Points &points, const KeyBuffer &keys)
    -> void {
  visit_morton_partitions(keys, [&](std::size_t first, std::size_t last) {
    std::sort(points.begin() + std::ptrdiff_t(first),
              points.begin() + std::ptrdiff_t(last),
              [](const auto &a, const auto &b) {
                return a[0] < b[0] || (a[0] == b[0] && a[1] < b[1]);
              });
  });
}

} // namespace tf::topology::cdt

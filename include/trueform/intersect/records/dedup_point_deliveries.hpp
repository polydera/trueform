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
#include "./point_delivery.hpp"
#include "tbb/parallel_sort.h"

#include <algorithm>

namespace tf::intersect {

/// Sort by face then point and collapse repeats. The resulting order is
/// canonical, its leading key is `key()`, and within a face the point ids
/// ascend — so a later grouping pass needs no sort of its own and a
/// face's block is the sorted set the fan's gate merges.
template <typename Index>
auto dedup_point_deliveries(tf::buffer<point_delivery<Index>> &deliveries)
    -> void {
  if (deliveries.size() == 0)
    return;
  // A small batch is sorted below the cost of entering the parallel
  // machinery.
  if (deliveries.size() < 4096)
    std::sort(deliveries.begin(), deliveries.end());
  else
    tbb::parallel_sort(deliveries.begin(), deliveries.end());
  deliveries.erase_till_end(
      std::unique(deliveries.begin(), deliveries.end()));
}

} // namespace tf::intersect

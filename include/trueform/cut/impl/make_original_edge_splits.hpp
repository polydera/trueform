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

namespace tf::cut {

template <typename Index, typename Split, typename OriginalEdgeSplit>
auto make_original_edge_splits(
    const tf::buffer<Split> &splits, Index created_tag,
    tf::buffer<OriginalEdgeSplit> &original_splits) -> void {
  original_splits.clear();
  for (const auto &split : splits) {
    const auto &a = split.edge[0];
    const auto &b = split.edge[1];
    if (a[0] == created_tag || b[0] == created_tag || a[0] != b[0])
      continue;
    original_splits.push_back({a[0], a[1], b[1], split.vertex});
  }
}

} // namespace tf::cut

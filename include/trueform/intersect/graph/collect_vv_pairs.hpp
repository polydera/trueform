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

#include "../../core/buffer.hpp"
#include "../../core/reallocate.hpp"

#include <algorithm>
#include <array>

namespace tf::intersect::graph {

/// Collect unique VV merge pairs from sorted VV crossing records.
///
/// Deduplicates vv_range in-place, appends {point_a, point_b} pairs
/// into the caller's merge_pairs buffer.
template <typename Index, typename VVRange>
auto collect_vv_pairs(VVRange &&vv_range,
                      tf::buffer<std::array<Index, 2>> &merge_pairs) -> void {
  auto vv_unique_end =
      std::unique(vv_range.begin(), vv_range.end(),
                  [](const auto &a, const auto &b) {
                    return a.point_a == b.point_a && a.point_b == b.point_b;
                  });
  auto n = static_cast<std::size_t>(vv_unique_end - vv_range.begin());
  if (n == 0)
    return;
  auto old_size = merge_pairs.size();
  merge_pairs.reallocate(old_size + n);
  auto out = merge_pairs.begin() + old_size;
  for (auto it = vv_range.begin(); it != vv_unique_end; ++it)
    *out++ = std::array<Index, 2>{it->point_a, it->point_b};
}

} // namespace tf::intersect::graph

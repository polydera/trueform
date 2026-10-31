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
#include "../../core/config/verify.hpp"
#include "../../exact/meta.hpp"
#include "./plane_identity_collapse.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::intersect::graph {

/// Every split survivor speaks the closed `merges` and stands in order along
/// its root. `position(tag, id)` reads an identity, a created one at tag `-1`.
template <typename Index, typename Int, typename Graph, typename Position>
auto assert_split_survivors_are_ordered(
    const Graph &graph, const tf::buffer<std::array<Index, 3>> &merges,
    const tf::buffer<Index> &split_edge,
    const tf::buffer<Index> &split_offsets,
    const tf::buffer<Index> &split_data, const Position &position) -> void {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  for (std::size_t group = 0; group < split_edge.size(); ++group) {
    const auto &root = graph.canon_group(split_edge[group])[0];
    const auto lo = position(root.point_tag_0, root.point_0);
    const auto hi = position(root.point_tag_1, root.point_1);
    T2 previous = T2(0);
    for (auto at = split_offsets[group]; at < split_offsets[group + 1]; ++at) {
      const auto id = split_data[std::size_t(at)];
      tf::core::verify(plane_merge_target(merges, Index(1), id) == Index(-1),
                       "a split survivor speaks the closed rewrite table");
      const auto p = position(std::int16_t(-1), id);
      T2 key = T2(0);
      for (int k = 0; k < 3; ++k)
        key += T2(T1(hi[k]) - T1(lo[k])) * T2(T1(p[k]) - T1(lo[k]));
      tf::core::verify(key >= previous,
                       "a split's survivors stand in order along its root");
      previous = key;
    }
  }
}

} // namespace tf::intersect::graph

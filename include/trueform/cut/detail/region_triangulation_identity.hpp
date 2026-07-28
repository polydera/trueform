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
#include "../../intersect/graph/vertex.hpp"
#include <algorithm>
#include <array>
#include <utility>

namespace tf::cut::detail {

template <typename Index>
auto region_vertex_key(Index n_tags, Index tag,
                       const tf::intersect::graph::vertex<Index> &vertex)
    -> std::array<Index, 2> {
  return vertex.source == tf::intersect::graph::vertex_source::created
             ? std::array<Index, 2>{n_tags, vertex.id}
             : std::array<Index, 2>{tag, vertex.id};
}

template <typename Index>
auto region_key_is_created(Index n_tags,
                           const std::array<Index, 2> &key) -> bool {
  return key[0] == n_tags;
}

template <typename Index>
auto region_edge_key(const std::array<Index, 2> &a,
                     const std::array<Index, 2> &b)
    -> std::array<std::array<Index, 2>, 2> {
  return a < b ? std::array<std::array<Index, 2>, 2>{a, b}
               : std::array<std::array<Index, 2>, 2>{b, a};
}

template <typename Index, typename Split>
auto find_region_edge_splits(
    const std::array<std::array<Index, 2>, 2> &key,
    const tf::buffer<Split> &splits) -> std::pair<Index, Index> {
  struct compare {
    auto operator()(
        const Split &split,
        const std::array<std::array<Index, 2>, 2> &candidate) const -> bool {
      return split.edge < candidate;
    }
    auto operator()(
        const std::array<std::array<Index, 2>, 2> &candidate,
        const Split &split) const -> bool {
      return candidate < split.edge;
    }
  };
  auto range = std::equal_range(splits.begin(), splits.end(), key, compare{});
  return {Index(range.first - splits.begin()),
          Index(range.second - splits.begin())};
}

} // namespace tf::cut::detail

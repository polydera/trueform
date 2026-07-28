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
#include "../../core/offset_block_buffer.hpp"
#include "../detail/region_triangulation_identity.hpp"
#include "../detail/region_triangulation_types.hpp"
#include "../face_regions.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::cut {

template <typename Index, typename Int, typename Walk>
auto count_region_walk_splits(
    Index n_tags, Index tag, const Walk &walk,
    const tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>>
        &splits) -> Index {
  Index count = 0;
  const std::size_t size = walk.size();
  for (std::size_t i = 0, j = size - 1; i < size; j = i++) {
    const auto range = tf::cut::detail::find_region_edge_splits(
        tf::cut::detail::region_edge_key(
            tf::cut::detail::region_vertex_key(n_tags, tag, walk[j]),
            tf::cut::detail::region_vertex_key(n_tags, tag, walk[i])),
        splits);
    count += range.second - range.first;
  }
  return count;
}

template <typename Index>
auto count_region_steiners(
    Index region, const tf::offset_block_buffer<Index, Index> &region_steiners)
    -> Index {
  if (region_steiners.size() == 0)
    return Index(0);
  return Index(region_steiners[std::size_t(region)].size());
}

template <typename Index, typename Int>
auto region_refinement_fingerprint(
    const tf::face_regions<Index, Int> &regions, Index region, Index n_tags,
    const tf::buffer<tf::cut::detail::region_triangulation_split<Index, Int>>
        &splits,
    const tf::offset_block_buffer<Index, Index> &region_steiners)
    -> std::array<Index, 2> {
  const auto descriptor = regions.descriptors()[region];
  Index split_count = tf::cut::count_region_walk_splits(
      n_tags, descriptor.tag, regions.loops()[region], splits);
  for (auto hole : regions.loop_holes()[region])
    split_count += tf::cut::count_region_walk_splits(
        n_tags, descriptor.tag, regions.holes()[hole], splits);
  return {split_count,
          tf::cut::count_region_steiners(region, region_steiners)};
}

} // namespace tf::cut

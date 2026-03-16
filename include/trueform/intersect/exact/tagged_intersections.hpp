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

#include "../../core/algorithm/compute_offsets.hpp"
#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "./tagged_intersection.hpp"
#include "tbb/parallel_sort.h"

namespace tf::exact {

template <typename Index, typename RealType, std::size_t Dims>
class tagged_intersections {
public:
  /// All intersections grouped by key() = (tag, object).
  auto intersections() const {
    return tf::make_offset_block_range(_intersections_offsets, _intersections);
  }

  /// Intersections for mesh `tag` — all groups where tag == i.
  /// Returns an offset_block_range over the slice of groups for that tag.
  auto intersections(Index tag) const {
    auto off_begin = _tag_offsets[tag];
    auto off_end = _tag_offsets[tag + 1];
    auto offsets_slice =
        tf::make_range(_intersections_offsets.begin() + off_begin,
                        _intersections_offsets.begin() + off_end + 1);
    return tf::make_offset_block_range(offsets_slice, _intersections);
  }

  auto intersection_points() const {
    return tf::make_range(_intersection_points);
  }

  auto flat_intersections() const { return tf::make_range(_intersections); }

  auto clear() {
    _intersections.clear();
    _intersections_offsets.clear();
    _tag_offsets.clear();
    _intersection_points.clear();
  }

  auto get_flat_index(const tagged_intersection<Index> &i) const -> Index {
    return &i - _intersections.begin();
  }

protected:
  /// Sort intersections, compute group offsets by key(), and tag-level offsets.
  /// `n_tags` is the number of meshes.
  auto finalize(Index n_ids, Index n_tags) {
    if (n_ids == 0)
      return;
    tbb::parallel_sort(_intersections.begin(), _intersections.end());

    // Group offsets by key() = (tag, object)
    _intersections_offsets.reserve(n_ids * 2 + 1);
    tf::compute_offsets(_intersections,
                        std::back_inserter(_intersections_offsets), Index(0),
                        [](const auto &x0, const auto &x1) {
                          return x0.key() == x1.key();
                        });

    // Tag-level offsets: _tag_offsets[i] = first group index where tag >= i
    auto n_groups = _intersections_offsets.size() - 1;
    _tag_offsets.allocate(n_tags + 1);
    _tag_offsets[0] = Index(0);

    Index g = 0;
    for (Index t = 0; t < n_tags; ++t) {
      while (g < Index(n_groups) &&
             _intersections[_intersections_offsets[g]].tag == t)
        ++g;
      _tag_offsets[t + 1] = g;
    }
  }

  tf::buffer<tagged_intersection<Index>> _intersections;
  tf::buffer<Index> _intersections_offsets;
  tf::buffer<Index> _tag_offsets;
  tf::buffer<tf::point<RealType, Dims>> _intersection_points;
};

} // namespace tf::exact

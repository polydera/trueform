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
#include "../../core/views/drop.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/take.hpp"
#include "./tagged_intersection.hpp"
#include "tbb/parallel_sort.h"

namespace tf::intersect {

template <typename Index, typename RealType, std::size_t Dims>
class tagged_intersections {
public:
  auto intersections() const {
    return tf::make_offset_block_range(_intersections_offsets, _intersections);
  }

  auto intersections0() const {
    return tf::take(intersections(), _partition_id);
  }

  auto intersections1() const {
    return tf::drop(intersections(), _partition_id);
  }

  auto intersection_points() const {
    return tf::make_range(_intersection_points);
  }

  auto flat_intersections() const {
    return tf::make_range(_intersections);
  }

  auto clear() {
    _intersections.clear();
    _intersections_offsets.clear();
    _intersection_points.clear();
    _partition_id = 0;
  }

  auto get_flat_index(const intersect::tagged_intersection<Index> &i) const
      -> Index {
    return &i - _intersections.begin();
  }

protected:
  auto finalize(Index n_ids) {
    if (n_ids == 0)
      return;
    tbb::parallel_sort(_intersections.begin(), _intersections.end());
    _intersections_offsets.reserve(n_ids * 2 + 1);
    tf::compute_offsets(_intersections,
                        std::back_inserter(_intersections_offsets), Index(0),
                        [](const auto &x0, const auto &x1) {
                          return x0.object_key() == x1.object_key();
                        });
    auto r = tf::make_indirect_range(
        tf::make_range(_intersections_offsets.begin(),
                       _intersections_offsets.size() - 1),
        _intersections);
    _partition_id = std::upper_bound(r.begin(), r.end(), 0,
                                     [](const auto &value, const auto &r1) {
                                       return value < r1.tag;
                                     }) -
                    r.begin();
  }

  Index _partition_id = 0;
  tf::buffer<intersect::tagged_intersection<Index>> _intersections;
  tf::buffer<Index> _intersections_offsets;
  tf::buffer<tf::point<RealType, Dims>> _intersection_points;
};
} // namespace tf::intersect

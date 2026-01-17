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
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/policy/ids.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./simple_edge_point_id.hpp"
#include "./simple_intersection.hpp"
#include "tbb/parallel_sort.h"

namespace tf::intersect {
template <typename Index, typename RealType, std::size_t Dims>
class simple_intersections {
public:
  auto clear() {
    _points.clear();
    _intersections.clear();
    _intersections_offsets.clear();
    _vertex_points_offset = 0;
  }

  auto intersections() const {
    return tf::make_offset_block_range(_intersections_offsets, _intersections);
  }

  auto intersection_points() const { return tf::make_range(_points); }

  auto created_intersection_points() const {
    return tf::make_range(_points.begin(),
                          _points.begin() + _vertex_points_offset);
  }

  auto flat_intersections() const { return tf::make_range(_intersections); }

  auto get_flat_index(const intersect::simple_intersection<Index> &i) const
      -> Index {
    return &i - _intersections.begin();
  }

private:
  auto merge_points(tf::buffer<simple_edge_point_id<Index>> &&edge_point_ids) {
    if (!edge_point_ids.size()) {
      _vertex_points_offset = 0;
      return;
    }
    tbb::parallel_sort(edge_point_ids);
    tf::buffer<Index> id_map;
    tf::buffer<tf::point<RealType, Dims>> new_points;
    new_points.reserve(_points.size());
    id_map.allocate(_points.size());
    auto iter = edge_point_ids.begin();
    auto end = edge_point_ids.end();
    Index current_id = 0;
    new_points.push_back(_points[iter->point_id]);
    id_map[iter->point_id] = current_id;
    auto check_with = iter;
    _vertex_points_offset = 0;
    while (++iter != end) {
      if (!(*check_with == *iter)) {
        ++current_id;
        _vertex_points_offset += (iter->vertex_id0 != iter->vertex_id1);
        check_with = iter;
        new_points.push_back(_points[iter->point_id]);
      }
      id_map[iter->point_id] = current_id;
    }
    ++_vertex_points_offset;
    tf::parallel_for_each(_intersections, [&](auto &intersection) {
      intersection.id = id_map[intersection.id];
    });
    _points = std::move(new_points);
  }

  auto compute_offsets() {
    tbb::parallel_sort(_intersections);
    _intersections.erase(
        std::unique(_intersections.begin(), _intersections.end()),
        _intersections.end());
    _intersections_offsets.reserve(_intersections.size() / 2 + 1);
    tf::compute_offsets(_intersections,
                        std::back_inserter(_intersections_offsets), 0,
                        [](const auto &x0, const auto &x1) {
                          return x0.object_key() == x1.object_key();
                        });
  }

protected:
  auto finalize(tf::buffer<simple_edge_point_id<Index>> &&edge_point_ids) {
    merge_points(std::move(edge_point_ids));
    compute_offsets();
  }

  tf::buffer<simple_intersection<Index>> _intersections;
  tf::buffer<Index> _intersections_offsets;
  tf::buffer<tf::point<RealType, Dims>> _points;
  Index _vertex_points_offset = 0;
};
} // namespace tf::intersect

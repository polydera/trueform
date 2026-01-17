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
#include "../clean/index_map/points.hpp"
#include "../core/algorithm/compute_offsets.hpp"
#include "../core/algorithm/generic_generate.hpp"
#include "../core/algorithm/mask_to_map.hpp"
#include "../core/algorithm/parallel_copy.hpp"
#include "../core/local_buffer.hpp"
#include "../core/views/offset_block_range.hpp"
#include "../spatial/search_self.hpp"
#include "../spatial/tree_like.hpp"
#include "../topology/policy/edge_membership.hpp"
#include "./detail/compute_simplification_mask.hpp"
#include "./detail/duplicate_intersection.hpp"
#include "./generate/segment_segment.hpp"

namespace tf {

/// @ingroup intersect_data
/// @brief Low-level intersection data within a 2D segment collection.
///
/// Computes and stores all intersection points between @ref tf::segments.
/// The segments must have @ref tf::edge_membership policy attached.
///
/// @tparam Index The index type.
/// @tparam RealT The coordinate type.
/// @tparam Dims The number of dimensions.
template <typename Index, typename RealT, std::size_t Dims>
class intersections_within_segments {
public:
  template <typename Policy, typename TreePolicy>
  auto build(const tf::segments<Policy> &segments,
             const tf::tree_like<TreePolicy> &tree) {
    static_assert(TreePolicy::coordinate_dims::value == Dims,
                  "Tree dimension mismatch");
    static_assert(tf::has_edge_membership_policy<Policy>,
                  "Use: segments | tf::tag(edge_membership)");
    clear();
    auto [intersections, points] =
        generate_initial_intersections(segments, tree);
    auto keep_mask =
        intersect::compute_simplification_mask(intersections, segments.edges());
    tf::buffer<Index> map;
    map.allocate(keep_mask.size());
    auto n_ids = tf::mask_to_map(keep_mask, map);
    _intersection_points.allocate(n_ids);
    auto fill_f = [&, &points = points,
                   none = Index(map.size())](auto intersection, auto &buffer) {
      if (map[intersection.id] == none)
        return;
      auto id = intersection.id;
      intersection.id = map[intersection.id];
      _intersection_points[intersection.id] = points[id];
      intersect::duplicate_intersection(intersection,
                                        segments.edge_membership(), buffer);
    };

    if (Index(intersections.size()) > 1000)
      tf::generic_generate(intersections, _intersections, fill_f);
    else {
      _intersections.reserve(n_ids * 2);
      for (auto intersection : intersections)
        fill_f(intersection, _intersections);
    }
    collapse_points(std::move(points));
    finalize(n_ids);
  }

  auto intersections() const {
    return tf::make_offset_block_range(_intersections_offsets, _intersections);
  }

  auto intersection_points() const {
    return tf::make_points(_intersection_points);
  }

  auto clear() {
    _intersection_points.clear();
    _intersections_offsets.clear();
    _intersections.clear();
  }

private:
  auto collapse_points(tf::buffer<tf::point<RealT, Dims>> &&points) {
    auto im = tf::make_clean_index_map<Index>(
        tf::make_points(_intersection_points), tf::epsilon<RealT>);
    if (im.kept_ids().size() == _intersection_points.size())
      return;
    points.allocate(im.kept_ids().size());
    tf::parallel_copy(
        tf::make_indirect_range(im.kept_ids(), _intersection_points), points);
    _intersection_points = std::move(points);
    tf::parallel_for_each(_intersections, [&](auto &i) { i.id = im.f()[i.id]; });
  }

  auto finalize(Index n_ids) {
    if (n_ids == 0)
      return;
    tbb::parallel_sort(_intersections.begin(), _intersections.end());
    _intersections_offsets.reserve(n_ids * 2 + 1);
    tf::compute_offsets(
        _intersections, std::back_inserter(_intersections_offsets), Index(0),
        [](const auto &x0, const auto &x1) { return x0.object == x1.object; });
  }

  template <typename Policy, typename TreePolicy>
  auto
  generate_initial_intersections(const tf::segments<Policy> &segments,
                                 const tf::tree_like<TreePolicy> &tree) {
    if (segments.size() < 1000)
      return generate_initial_intersections_seq(segments, tree);
    else
      return generate_initial_intersections_par(segments, tree);
  }

  template <typename Policy, typename TreePolicy>
  auto generate_initial_intersections_seq(
      const tf::segments<Policy> &segments,
      const tf::tree_like<TreePolicy> &tree) {
    tf::buffer<intersect::intersection<Index>> _intersections;
    tf::buffer<tf::point<RealT, Dims>> _points;
    tf::search_self(
        tree, tf::intersects_f,
        [&](Index id0, Index id1) {
          intersect::generate::segment_segment(
              segments.edge_membership(), segments[id0] | tf::tag_id(id0),
              segments[id1] | tf::tag_id(id1), _intersections, _points);
        },
        0);
    return std::make_pair(std::move(_intersections), std::move(_points));
  }

  template <typename Policy, typename TreePolicy>
  auto generate_initial_intersections_par(
      const tf::segments<Policy> &segments,
      const tf::tree_like<TreePolicy> &tree) {
    tf::local_buffer<intersect::intersection<Index>> l_intersections;
    tf::local_buffer<tf::point<RealT, Dims>> l_points;
    tf::search_self(tree, tf::intersects_f, [&](Index id0, Index id1) {
      intersect::generate::segment_segment(
          segments.edge_membership(), segments[id0] | tf::tag_id(id0),
          segments[id1] | tf::tag_id(id1), *l_intersections, *l_points);
    });

    tf::buffer<intersect::intersection<Index>> _intersections;
    tf::buffer<tf::point<RealT, Dims>> _points;
    l_points.to_buffer(_points);
    _intersections.allocate(l_intersections.total_size());
    auto it = _intersections.begin();
    std::size_t offset = 0;
    for (const auto &v : l_intersections.buffers()) {
      for (auto e : v) {
        e.id += offset;
        *it++ = e;
      }
      offset += v.size();
    }
    return std::make_pair(std::move(_intersections), std::move(_points));
  }

  tf::buffer<Index> _intersections_offsets;
  tf::buffer<intersect::intersection<Index>> _intersections;
  tf::buffer<tf::point<RealT, Dims>> _intersection_points;
};
} // namespace tf

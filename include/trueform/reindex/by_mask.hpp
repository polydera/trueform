/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/algorithm/mask_to_index_map.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "./points.hpp"
#include "./polygons.hpp"
#include "./return_index_map.hpp"
#include "./segments.hpp"

namespace tf {
template <typename Index, typename Policy, typename Range>
auto reindexed_by_mask(const tf::points<Policy> &points, const Range &mask,
                       tf::return_index_map_t) {
  auto im = tf::mask_to_index_map<Index>(mask);
  auto out = tf::reindexed(points, im);
  return std::make_pair(std::move(out), std::move(im));
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_mask(const tf::points<Policy> &points, const Range &mask) {
  auto im = tf::mask_to_index_map<Index>(mask);
  return tf::reindexed(points, im);
}

template <typename Policy, typename Range>
auto reindexed_by_mask(const tf::points<Policy> &points, const Range &mask,
                       tf::return_index_map_t) {
  using index_t = typename Policy::size_type;
  return tf::reindexed_by_mask<index_t>(points, mask, tf::return_index_map);
}

template <typename Policy, typename Range>
auto reindexed_by_mask(const tf::points<Policy> &points, const Range &mask) {
  using index_t = typename Policy::size_type;
  return tf::reindexed_by_mask<index_t>(points, mask);
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_mask(const tf::polygons<Policy> &polygons, const Range &mask,
                       tf::return_index_map_t) {
  auto face_im = tf::mask_to_index_map<Index>(mask);
  tf::buffer<bool> point_mask;
  point_mask.allocate(polygons.points().size());
  tf::parallel_fill(point_mask, false);
  tf::parallel_apply(
      tf::make_indirect_range(face_im.kept_ids(), polygons.faces()),
      [&](auto &&face) {
        for (auto &e : face)
          point_mask[e] = true;
      });
  auto point_im = tf::mask_to_index_map<Index>(point_mask);
  auto out = tf::reindexed(polygons, face_im, point_im);
  return std::make_tuple(std::move(out), std::move(face_im),
                         std::move(point_im));
}

template <typename Policy, typename Range>
auto reindexed_by_mask(const tf::polygons<Policy> &polygons, const Range &mask,
                       tf::return_index_map_t) {
  using index_t = std::decay_t<decltype(polygons.faces()[0][0])>;
  return tf::reindexed_by_mask<index_t>(polygons, mask, tf::return_index_map);
}

template <typename Policy, typename Range>
auto reindexed_by_mask(const tf::polygons<Policy> &polygons,
                       const Range &mask) {
  using index_t = std::decay_t<decltype(polygons.faces()[0][0])>;
  return tf::reindexed_by_mask<index_t>(polygons, mask);
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_mask(const tf::polygons<Policy> &polygons,
                       const Range &mask) {
  return std::get<0>(reindexed_by_mask(polygons, mask, tf::return_index_map));
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_mask(const tf::segments<Policy> &segments, const Range &mask,
                       tf::return_index_map_t) {
  auto edge_im = tf::mask_to_index_map<Index>(mask);
  tf::buffer<bool> point_mask;
  point_mask.allocate(segments.points().size());
  tf::parallel_fill(point_mask, false);
  tf::parallel_apply(
      tf::make_indirect_range(edge_im.kept_ids(), segments.edges()),
      [&](auto &&edge) {
        for (auto &e : edge)
          point_mask[e] = true;
      });
  auto point_im = tf::mask_to_index_map<Index>(point_mask);
  auto out = tf::reindexed(segments, edge_im, point_im);
  return std::make_tuple(std::move(out), std::move(edge_im),
                         std::move(point_im));
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_mask(const tf::segments<Policy> &segments,
                       const Range &mask) {
  return std::get<0>(reindexed_by_mask(segments, mask, tf::return_index_map));
}

template <typename Policy, typename Range>
auto reindexed_by_mask(const tf::segments<Policy> &segments, const Range &mask,
                       tf::return_index_map_t) {
  using index_t = std::decay_t<decltype(segments.edges()[0][0])>;
  return tf::reindexed_by_mask<index_t>(segments, mask, tf::return_index_map);
}

template <typename Policy, typename Range>
auto reindexed_by_mask(const tf::segments<Policy> &segments,
                       const Range &mask) {
  using index_t = std::decay_t<decltype(segments.edges()[0][0])>;
  return tf::reindexed_by_mask<index_t>(segments, mask);
}
} // namespace tf

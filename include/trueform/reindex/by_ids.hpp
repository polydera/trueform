/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/algorithm/ids_to_index_map.hpp"
#include "../core/algorithm/mask_to_index_map.hpp"
#include "./points.hpp"
#include "./polygons.hpp"
#include "./range.hpp"
#include "./return_index_map.hpp"
#include "./segments.hpp"
#include "./unit_vectors.hpp"
#include "./vectors.hpp"

namespace tf {
template <typename Index, typename Policy, typename Range>
auto reindexed_by_ids(const tf::polygons<Policy> &polygons, const Range &ids,
                      tf::return_index_map_t) {
  auto face_im = tf::ids_to_index_map<Index>(ids, polygons.faces().size());
  tf::buffer<bool> point_mask;
  point_mask.allocate(polygons.points().size());
  tf::parallel_fill(point_mask, false);
  // benign race: multiple threads may write `true` to the same byte.
  // safe because writes are idempotent and there’s a barrier at loop end.
  tf::parallel_apply(
      tf::make_indirect_range(face_im.kept_ids(), polygons.faces()),
      [&](auto &&face) {
        for (auto &e : face)
          point_mask[e] = true;
      },
      tf::checked);
  auto point_im = tf::mask_to_index_map<Index>(point_mask);
  auto out = tf::reindexed(polygons, face_im, point_im);
  return std::make_tuple(std::move(out), std::move(face_im),
                         std::move(point_im));
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_ids(const tf::polygons<Policy> &polygons, const Range &ids) {
  return std::get<0>(
      reindexed_by_ids<Index>(polygons, ids, tf::return_index_map));
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_ids(const tf::segments<Policy> &segments, const Range &ids,
                      tf::return_index_map_t) {
  auto edge_im = tf::ids_to_index_map<Index>(ids, segments.edges().size());
  tf::buffer<bool> point_mask;
  point_mask.allocate(segments.points().size());
  tf::parallel_fill(point_mask, false);
  // benign race: multiple threads may write `true` to the same byte.
  // safe because writes are idempotent and there’s a barrier at loop end.
  tf::parallel_apply(
      tf::make_indirect_range(edge_im.kept_ids(), segments.edges()),
      [&](auto &&edge) {
        for (auto &e : edge)
          point_mask[e] = true;
      },
      tf::checked);
  auto point_im = tf::mask_to_index_map<Index>(point_mask);
  auto out = tf::reindexed(segments, edge_im, point_im);
  return std::make_tuple(std::move(out), std::move(edge_im),
                         std::move(point_im));
}


template <typename Index, typename Policy, typename Range>
auto reindexed_by_ids(const tf::segments<Policy> &segments, const Range &ids) {
  return std::get<0>(
      reindexed_by_ids<Index>(segments, ids, tf::return_index_map));
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_ids(const tf::points<Policy> &points, const Range &ids,
                      tf::return_index_map_t) {
  auto im = tf::ids_to_index_map<Index>(ids, points.size());
  auto out = tf::reindexed(points, im);
  return std::make_pair(std::move(out), std::move(im));
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_ids(const tf::points<Policy> &points, const Range &ids) {
  return std::get<0>(
      reindexed_by_ids<Index>(points, ids, tf::return_index_map));
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_ids(const tf::vectors<Policy> &vectors, const Range &ids,
                      tf::return_index_map_t) {
  auto im = tf::ids_to_index_map<Index>(ids, vectors.size());
  auto out = tf::reindexed(vectors, im);
  return std::make_pair(std::move(out), std::move(im));
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_ids(const tf::vectors<Policy> &vectors, const Range &ids) {
  return std::get<0>(
      reindexed_by_ids<Index>(vectors, ids, tf::return_index_map));
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_ids(const tf::unit_vectors<Policy> &unit_vectors,
                      const Range &ids, tf::return_index_map_t) {
  auto im = tf::ids_to_index_map<Index>(ids, unit_vectors.size());
  auto out = tf::reindexed(unit_vectors, im);
  return std::make_pair(std::move(out), std::move(im));
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_ids(const tf::unit_vectors<Policy> &unit_vectors,
                      const Range &ids) {
  return std::get<0>(
      reindexed_by_ids<Index>(unit_vectors, ids, tf::return_index_map));
}


template <typename Index, typename Iter, std::size_t N, typename Range>
auto reindexed_by_ids(const tf::range<Iter, N> &r, const Range &ids,
                      tf::return_index_map_t) {
  auto im = tf::ids_to_index_map<Index>(ids, r.size());
  auto out = tf::reindexed(r, im);
  return std::make_pair(std::move(out), std::move(im));
}

template <typename Index, typename Iter, std::size_t N, typename Range>
auto reindexed_by_ids(const tf::range<Iter, N> &r, const Range &ids) {
  return std::get<0>(tf::reindexed_by_ids<Index>(r, ids, tf::return_index_map));
}
} // namespace tf

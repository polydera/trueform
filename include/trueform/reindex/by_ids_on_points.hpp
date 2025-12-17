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
#include "./return_index_map.hpp"
#include "./segments.hpp"

namespace tf {

template <typename Index, typename Policy, typename Range>
auto reindexed_by_ids_on_points(const tf::polygons<Policy> &polygons,
                                const Range &ids, tf::return_index_map_t) {
  // point mapping from ids (stable gather)
  auto point_im = tf::ids_to_index_map<Index>(ids, polygons.points().size());

  // face mask: keep face iff all its vertices are present in point_im
  tf::buffer<bool> face_mask;
  face_mask.allocate(polygons.faces().size());

  const Index none =
      Index(point_im.f().size()); // sentinel used in ids_to_index_map
  tf::parallel_apply(
      tf::zip(face_mask, polygons.faces()),
      [&](auto &&zipped) {
        auto &keep = std::get<0>(zipped);
        auto &&face = std::get<1>(zipped);

        unsigned char k = 1; // branchless all-of
        for (auto v : face)
          k &= static_cast<unsigned char>(point_im.f()[v] != none);
        keep = (k != 0);
      },
      tf::checked);

  auto face_im = tf::mask_to_index_map<Index>(face_mask);
  auto out = tf::reindexed(polygons, face_im, point_im);
  return std::make_tuple(std::move(out), std::move(face_im),
                         std::move(point_im));
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_ids_on_points(const tf::polygons<Policy> &polygons,
                                const Range &ids) {
  return std::get<0>(tf::reindexed_by_ids_on_points<Index>(
      polygons, ids, tf::return_index_map));
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_ids_on_points(const tf::segments<Policy> &segments,
                                const Range &ids, tf::return_index_map_t) {
  // point mapping from ids (stable gather)
  auto point_im = tf::ids_to_index_map<Index>(ids, segments.points().size());

  // edge mask: keep edge iff all its vertices are present in point_im
  tf::buffer<bool> edge_mask;
  edge_mask.allocate(segments.edges().size());

  const Index none =
      Index(point_im.f().size()); // sentinel used in ids_to_index_map
  tf::parallel_apply(
      tf::zip(edge_mask, segments.edges()),
      [&](auto &&zipped) {
        auto &keep = std::get<0>(zipped);
        auto &&edge = std::get<1>(zipped);

        unsigned char k = 1;
        for (auto v : edge)
          k &= static_cast<unsigned char>(point_im.f()[v] != none);
        keep = (k != 0);
      },
      tf::checked);

  auto edge_im = tf::mask_to_index_map<Index>(edge_mask);
  auto out = tf::reindexed(segments, edge_im, point_im);
  return std::make_tuple(std::move(out), std::move(edge_im),
                         std::move(point_im));
}

template <typename Index, typename Policy, typename Range>
auto reindexed_by_ids_on_points(const tf::segments<Policy> &segments,
                                const Range &ids) {
  return std::get<0>(tf::reindexed_by_ids_on_points<Index>(
      segments, ids, tf::return_index_map));
}

} // namespace tf

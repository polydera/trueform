/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include "../core/algorithm/mask_to_index_map.hpp"
#include "../core/none.hpp"
#include "./points.hpp"
#include "./polygons.hpp"
#include "./return_index_map.hpp"
#include "./segments.hpp"

namespace tf {

// =============================================================================
// polygons - deduces from faces()[0][0]
// =============================================================================

template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_mask_on_points(const tf::polygons<Policy> &polygons,
                                 const Range &point_mask,
                                 tf::return_index_map_t) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return reindexed_by_mask_on_points<ActualIndex>(polygons, point_mask,
                                                    tf::return_index_map);
  } else {
    auto point_im = tf::mask_to_index_map<Index>(point_mask);

    tf::buffer<bool> face_mask;
    face_mask.allocate(polygons.faces().size());

    tf::parallel_apply(
        tf::zip(face_mask, polygons.faces()),
        [&](auto &&zipped) {
          auto &keep = std::get<0>(zipped);
          auto &&face = std::get<1>(zipped);
          unsigned char k = 1;
          for (auto v : face)
            k &= static_cast<unsigned char>(point_mask[v]);
          keep = (k != 0);
        },
        tf::checked);

    auto face_im = tf::mask_to_index_map<Index>(face_mask);
    auto out = tf::reindexed(polygons, face_im, point_im);
    return std::make_tuple(std::move(out), std::move(face_im),
                           std::move(point_im));
  }
}

template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_mask_on_points(const tf::polygons<Policy> &polygons,
                                 const Range &point_mask) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return reindexed_by_mask_on_points<ActualIndex>(polygons, point_mask);
  } else {
    return std::get<0>(tf::reindexed_by_mask_on_points<Index>(
        polygons, point_mask, tf::return_index_map));
  }
}

// =============================================================================
// segments - deduces from edges()[0][0]
// =============================================================================

template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_mask_on_points(const tf::segments<Policy> &segments,
                                 const Range &point_mask,
                                 tf::return_index_map_t) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return reindexed_by_mask_on_points<ActualIndex>(segments, point_mask,
                                                    tf::return_index_map);
  } else {
    auto point_im = tf::mask_to_index_map<Index>(point_mask);

    tf::buffer<bool> edge_mask;
    edge_mask.allocate(segments.edges().size());

    tf::parallel_apply(
        tf::zip(edge_mask, segments.edges()),
        [&](auto &&zipped) {
          auto &keep = std::get<0>(zipped);
          auto &&edge = std::get<1>(zipped);

          unsigned char k = 1;
          for (auto v : edge)
            k &= static_cast<unsigned char>(point_mask[v]);
          keep = (k != 0);
        },
        tf::checked);

    auto edge_im = tf::mask_to_index_map<Index>(edge_mask);
    auto out = tf::reindexed(segments, edge_im, point_im);
    return std::make_tuple(std::move(out), std::move(edge_im),
                           std::move(point_im));
  }
}

template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_mask_on_points(const tf::segments<Policy> &segments,
                                 const Range &point_mask) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return reindexed_by_mask_on_points<ActualIndex>(segments, point_mask);
  } else {
    return std::get<0>(tf::reindexed_by_mask_on_points<Index>(
        segments, point_mask, tf::return_index_map));
  }
}

} // namespace tf

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

#include "../core/algorithm/ids_to_index_map.hpp"
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

/// @ingroup reindex
/// @brief Filter polygons by point IDs and return index maps.
///
/// Keeps only faces whose all vertices are in the point ID list.
/// Derives face mask from point selection.
///
/// @tparam Index The index type (auto-deduced from geometry if not specified).
/// @tparam Policy The policy type of the polygons.
/// @tparam Range ID range type.
/// @param polygons The input @ref tf::polygons.
/// @param ids Range of point indices to keep.
/// @param tag Pass @ref tf::return_index_map to get the mappings.
/// @return Tuple of (@ref tf::polygons_buffer, face @ref tf::index_map_buffer, point @ref tf::index_map_buffer).
template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_ids_on_points(const tf::polygons<Policy> &polygons,
                                const Range &ids, tf::return_index_map_t) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return reindexed_by_ids_on_points<ActualIndex>(polygons, ids,
                                                   tf::return_index_map);
  } else {
    // point mapping from ids (stable gather)
    auto point_im = tf::ids_to_index_map<Index>(ids, polygons.points().size());

    // face mask: keep face iff all its vertices are present in point_im
    tf::buffer<bool> face_mask;
    face_mask.allocate(polygons.faces().size());

    const Index none =
        Index(point_im.f().size()); // sentinel used in ids_to_index_map
    tf::parallel_for_each(
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
}

/// @ingroup reindex
/// @brief Filter polygons by point IDs.
/// @overload
template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_ids_on_points(const tf::polygons<Policy> &polygons,
                                const Range &ids) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return reindexed_by_ids_on_points<ActualIndex>(polygons, ids);
  } else {
    return std::get<0>(tf::reindexed_by_ids_on_points<Index>(
        polygons, ids, tf::return_index_map));
  }
}

// =============================================================================
// segments - deduces from edges()[0][0]
// =============================================================================

/// @ingroup reindex
/// @brief Filter segments by point IDs and return index maps.
///
/// Keeps only edges whose all vertices are in the point ID list.
/// Derives edge mask from point selection.
///
/// @tparam Index The index type (auto-deduced from geometry if not specified).
/// @tparam Policy The policy type of the segments.
/// @tparam Range ID range type.
/// @param segments The input @ref tf::segments.
/// @param ids Range of point indices to keep.
/// @param tag Pass @ref tf::return_index_map to get the mappings.
/// @return Tuple of (@ref tf::segments_buffer, edge @ref tf::index_map_buffer, point @ref tf::index_map_buffer).
template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_ids_on_points(const tf::segments<Policy> &segments,
                                const Range &ids, tf::return_index_map_t) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return reindexed_by_ids_on_points<ActualIndex>(segments, ids,
                                                   tf::return_index_map);
  } else {
    // point mapping from ids (stable gather)
    auto point_im = tf::ids_to_index_map<Index>(ids, segments.points().size());

    // edge mask: keep edge iff all its vertices are present in point_im
    tf::buffer<bool> edge_mask;
    edge_mask.allocate(segments.edges().size());

    const Index none =
        Index(point_im.f().size()); // sentinel used in ids_to_index_map
    tf::parallel_for_each(
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
}

/// @ingroup reindex
/// @brief Filter segments by point IDs.
/// @overload
template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_ids_on_points(const tf::segments<Policy> &segments,
                                const Range &ids) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return reindexed_by_ids_on_points<ActualIndex>(segments, ids);
  } else {
    return std::get<0>(tf::reindexed_by_ids_on_points<Index>(
        segments, ids, tf::return_index_map));
  }
}

} // namespace tf

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
/// @brief Filter polygons by point mask and return index maps.
///
/// Keeps only faces whose all vertices pass the point mask.
/// Derives face mask from point selection.
///
/// @tparam Index The index type (auto-deduced from geometry if not specified).
/// @tparam Policy The policy type of the polygons.
/// @tparam Range Mask range type.
/// @param polygons The input @ref tf::polygons.
/// @param point_mask Boolean range for points.
/// @param tag Pass @ref tf::return_index_map to get the mappings.
/// @return Tuple of (@ref tf::polygons_buffer, face @ref tf::index_map_buffer, point @ref tf::index_map_buffer).
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

    tf::parallel_for_each(
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

/// @ingroup reindex
/// @brief Filter polygons by point mask.
/// @overload
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

/// @ingroup reindex
/// @brief Filter segments by point mask and return index maps.
///
/// Keeps only edges whose all vertices pass the point mask.
/// Derives edge mask from point selection.
///
/// @tparam Index The index type (auto-deduced from geometry if not specified).
/// @tparam Policy The policy type of the segments.
/// @tparam Range Mask range type.
/// @param segments The input @ref tf::segments.
/// @param point_mask Boolean range for points.
/// @param tag Pass @ref tf::return_index_map to get the mappings.
/// @return Tuple of (@ref tf::segments_buffer, edge @ref tf::index_map_buffer, point @ref tf::index_map_buffer).
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

    tf::parallel_for_each(
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

/// @ingroup reindex
/// @brief Filter segments by point mask.
/// @overload
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

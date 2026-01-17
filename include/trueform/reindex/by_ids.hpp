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
#include "./range.hpp"
#include "./return_index_map.hpp"
#include "./segments.hpp"
#include "./unit_vectors.hpp"
#include "./vectors.hpp"

namespace tf {

// =============================================================================
// polygons - deduces from faces()[0][0]
// =============================================================================

/// @ingroup reindex
/// @brief Filter polygons by face IDs and return index maps.
///
/// Keeps only faces at specified indices. Automatically removes unreferenced points.
///
/// @tparam Index The index type (auto-deduced from geometry if not specified).
/// @tparam Policy The policy type of the polygons.
/// @tparam Range ID range type.
/// @param polygons The input @ref tf::polygons.
/// @param ids Range of face indices to keep.
/// @param tag Pass @ref tf::return_index_map to get the mappings.
/// @return Tuple of (@ref tf::polygons_buffer, face @ref tf::index_map_buffer, point @ref tf::index_map_buffer).
template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_ids(const tf::polygons<Policy> &polygons, const Range &ids,
                      tf::return_index_map_t) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return reindexed_by_ids<ActualIndex>(polygons, ids, tf::return_index_map);
  } else {
    auto face_im = tf::ids_to_index_map<Index>(ids, polygons.faces().size());
    tf::buffer<bool> point_mask;
    point_mask.allocate(polygons.points().size());
    tf::parallel_fill(point_mask, false);
    // benign race: multiple threads may write `true` to the same byte.
    // safe because writes are idempotent and there's a barrier at loop end.
    tf::parallel_for_each(
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
}

/// @ingroup reindex
/// @brief Filter polygons by face IDs.
/// @overload
template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_ids(const tf::polygons<Policy> &polygons, const Range &ids) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return reindexed_by_ids<ActualIndex>(polygons, ids);
  } else {
    return std::get<0>(
        reindexed_by_ids<Index>(polygons, ids, tf::return_index_map));
  }
}

// =============================================================================
// segments - deduces from edges()[0][0]
// =============================================================================

/// @ingroup reindex
/// @brief Filter segments by edge IDs and return index maps.
///
/// Keeps only edges at specified indices. Automatically removes unreferenced points.
///
/// @tparam Index The index type (auto-deduced from geometry if not specified).
/// @tparam Policy The policy type of the segments.
/// @tparam Range ID range type.
/// @param segments The input @ref tf::segments.
/// @param ids Range of edge indices to keep.
/// @param tag Pass @ref tf::return_index_map to get the mappings.
/// @return Tuple of (@ref tf::segments_buffer, edge @ref tf::index_map_buffer, point @ref tf::index_map_buffer).
template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_ids(const tf::segments<Policy> &segments, const Range &ids,
                      tf::return_index_map_t) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return reindexed_by_ids<ActualIndex>(segments, ids, tf::return_index_map);
  } else {
    auto edge_im = tf::ids_to_index_map<Index>(ids, segments.edges().size());
    tf::buffer<bool> point_mask;
    point_mask.allocate(segments.points().size());
    tf::parallel_fill(point_mask, false);
    // benign race: multiple threads may write `true` to the same byte.
    // safe because writes are idempotent and there's a barrier at loop end.
    tf::parallel_for_each(
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
}

/// @ingroup reindex
/// @brief Filter segments by edge IDs.
/// @overload
template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_ids(const tf::segments<Policy> &segments, const Range &ids) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return reindexed_by_ids<ActualIndex>(segments, ids);
  } else {
    return std::get<0>(
        reindexed_by_ids<Index>(segments, ids, tf::return_index_map));
  }
}

// =============================================================================
// points - defaults to int
// =============================================================================

/// @ingroup reindex
/// @brief Filter points by IDs and return index map.
///
/// @tparam Index The index type (defaults to int).
/// @tparam Policy The policy type of the points.
/// @tparam Range ID range type.
/// @param points The input @ref tf::points.
/// @param ids Range of point indices to keep.
/// @param tag Pass @ref tf::return_index_map to get the mapping.
/// @return Pair of (@ref tf::points_buffer, @ref tf::index_map_buffer).
template <typename Index = int, typename Policy, typename Range>
auto reindexed_by_ids(const tf::points<Policy> &points, const Range &ids,
                      tf::return_index_map_t) {
  auto im = tf::ids_to_index_map<Index>(ids, points.size());
  auto out = tf::reindexed(points, im);
  return std::make_pair(std::move(out), std::move(im));
}

/// @ingroup reindex
/// @brief Filter points by IDs.
/// @overload
template <typename Index = int, typename Policy, typename Range>
auto reindexed_by_ids(const tf::points<Policy> &points, const Range &ids) {
  return std::get<0>(
      reindexed_by_ids<Index>(points, ids, tf::return_index_map));
}

// =============================================================================
// vectors - defaults to int
// =============================================================================

/// @ingroup reindex
/// @brief Filter vectors by IDs and return index map.
///
/// @tparam Index The index type (defaults to int).
/// @tparam Policy The policy type of the vectors.
/// @tparam Range ID range type.
/// @param vectors The input @ref tf::vectors.
/// @param ids Range of vector indices to keep.
/// @param tag Pass @ref tf::return_index_map to get the mapping.
/// @return Pair of (@ref tf::vectors_buffer, @ref tf::index_map_buffer).
template <typename Index = int, typename Policy, typename Range>
auto reindexed_by_ids(const tf::vectors<Policy> &vectors, const Range &ids,
                      tf::return_index_map_t) {
  auto im = tf::ids_to_index_map<Index>(ids, vectors.size());
  auto out = tf::reindexed(vectors, im);
  return std::make_pair(std::move(out), std::move(im));
}

/// @ingroup reindex
/// @brief Filter vectors by IDs.
/// @overload
template <typename Index = int, typename Policy, typename Range>
auto reindexed_by_ids(const tf::vectors<Policy> &vectors, const Range &ids) {
  return std::get<0>(
      reindexed_by_ids<Index>(vectors, ids, tf::return_index_map));
}

// =============================================================================
// unit_vectors - defaults to int
// =============================================================================

/// @ingroup reindex
/// @brief Filter unit vectors by IDs and return index map.
///
/// @tparam Index The index type (defaults to int).
/// @tparam Policy The policy type of the unit vectors.
/// @tparam Range ID range type.
/// @param unit_vectors The input @ref tf::unit_vectors.
/// @param ids Range of unit vector indices to keep.
/// @param tag Pass @ref tf::return_index_map to get the mapping.
/// @return Pair of (@ref tf::unit_vectors_buffer, @ref tf::index_map_buffer).
template <typename Index = int, typename Policy, typename Range>
auto reindexed_by_ids(const tf::unit_vectors<Policy> &unit_vectors,
                      const Range &ids, tf::return_index_map_t) {
  auto im = tf::ids_to_index_map<Index>(ids, unit_vectors.size());
  auto out = tf::reindexed(unit_vectors, im);
  return std::make_pair(std::move(out), std::move(im));
}

/// @ingroup reindex
/// @brief Filter unit vectors by IDs.
/// @overload
template <typename Index = int, typename Policy, typename Range>
auto reindexed_by_ids(const tf::unit_vectors<Policy> &unit_vectors,
                      const Range &ids) {
  return std::get<0>(
      reindexed_by_ids<Index>(unit_vectors, ids, tf::return_index_map));
}

// =============================================================================
// range - defaults to int
// =============================================================================

/// @ingroup reindex
/// @brief Filter range by IDs and return index map.
///
/// @tparam Index The index type (defaults to int).
/// @tparam Iter Input iterator type.
/// @tparam N Static size hint.
/// @tparam Range ID range type.
/// @param r The input @ref tf::range.
/// @param ids Range of element indices to keep.
/// @param tag Pass @ref tf::return_index_map to get the mapping.
/// @return Pair of (filtered buffer, @ref tf::index_map_buffer).
template <typename Index = int, typename Iter, std::size_t N, typename Range>
auto reindexed_by_ids(const tf::range<Iter, N> &r, const Range &ids,
                      tf::return_index_map_t) {
  auto im = tf::ids_to_index_map<Index>(ids, r.size());
  auto out = tf::reindexed(r, im);
  return std::make_pair(std::move(out), std::move(im));
}

/// @ingroup reindex
/// @brief Filter range by IDs.
/// @overload
template <typename Index = int, typename Iter, std::size_t N, typename Range>
auto reindexed_by_ids(const tf::range<Iter, N> &r, const Range &ids) {
  return std::get<0>(tf::reindexed_by_ids<Index>(r, ids, tf::return_index_map));
}

} // namespace tf

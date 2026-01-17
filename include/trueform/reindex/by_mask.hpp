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
#include "../core/algorithm/parallel_fill.hpp"
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
// range - defaults to int
// =============================================================================

/// @ingroup reindex
/// @brief Filter range by boolean mask and return index map.
///
/// Keeps elements where mask is true.
///
/// @tparam Index The index type (defaults to int).
/// @tparam Iter Input iterator type.
/// @tparam N Static size hint.
/// @tparam Range Mask range type.
/// @param r The input @ref tf::range.
/// @param mask Boolean range indicating which elements to keep.
/// @param tag Pass @ref tf::return_index_map to get the mapping.
/// @return Pair of (filtered buffer, @ref tf::index_map_buffer).
template <typename Index = int, typename Iter, std::size_t N, typename Range>
auto reindexed_by_mask(const tf::range<Iter, N> &r, const Range &mask,
                       tf::return_index_map_t) {
  auto im = tf::mask_to_index_map<Index>(mask);
  auto out = tf::reindexed(r, im);
  return std::make_pair(std::move(out), std::move(im));
}

/// @ingroup reindex
/// @brief Filter range by boolean mask.
/// @overload
template <typename Index = int, typename Iter, std::size_t N, typename Range>
auto reindexed_by_mask(const tf::range<Iter, N> &r, const Range &mask) {
  auto im = tf::mask_to_index_map<Index>(mask);
  return tf::reindexed(r, im);
}

// =============================================================================
// points - defaults to int
// =============================================================================

/// @ingroup reindex
/// @brief Filter points by boolean mask and return index map.
///
/// @tparam Index The index type (defaults to int).
/// @tparam Policy The policy type of the points.
/// @tparam Range Mask range type.
/// @param points The input @ref tf::points.
/// @param mask Boolean range indicating which points to keep.
/// @param tag Pass @ref tf::return_index_map to get the mapping.
/// @return Pair of (@ref tf::points_buffer, @ref tf::index_map_buffer).
template <typename Index = int, typename Policy, typename Range>
auto reindexed_by_mask(const tf::points<Policy> &points, const Range &mask,
                       tf::return_index_map_t) {
  auto im = tf::mask_to_index_map<Index>(mask);
  auto out = tf::reindexed(points, im);
  return std::make_pair(std::move(out), std::move(im));
}

/// @ingroup reindex
/// @brief Filter points by boolean mask.
/// @overload
template <typename Index = int, typename Policy, typename Range>
auto reindexed_by_mask(const tf::points<Policy> &points, const Range &mask) {
  auto im = tf::mask_to_index_map<Index>(mask);
  return tf::reindexed(points, im);
}

// =============================================================================
// vectors - defaults to int
// =============================================================================

/// @ingroup reindex
/// @brief Filter vectors by boolean mask and return index map.
///
/// @tparam Index The index type (defaults to int).
/// @tparam Policy The policy type of the vectors.
/// @tparam Range Mask range type.
/// @param vectors The input @ref tf::vectors.
/// @param mask Boolean range indicating which vectors to keep.
/// @param tag Pass @ref tf::return_index_map to get the mapping.
/// @return Pair of (@ref tf::vectors_buffer, @ref tf::index_map_buffer).
template <typename Index = int, typename Policy, typename Range>
auto reindexed_by_mask(const tf::vectors<Policy> &vectors, const Range &mask,
                       tf::return_index_map_t) {
  auto im = tf::mask_to_index_map<Index>(mask);
  auto out = tf::reindexed(vectors, im);
  return std::make_pair(std::move(out), std::move(im));
}

/// @ingroup reindex
/// @brief Filter vectors by boolean mask.
/// @overload
template <typename Index = int, typename Policy, typename Range>
auto reindexed_by_mask(const tf::vectors<Policy> &vectors, const Range &mask) {
  auto im = tf::mask_to_index_map<Index>(mask);
  return tf::reindexed(vectors, im);
}

// =============================================================================
// unit_vectors - defaults to int
// =============================================================================

/// @ingroup reindex
/// @brief Filter unit vectors by boolean mask and return index map.
///
/// @tparam Index The index type (defaults to int).
/// @tparam Policy The policy type of the unit vectors.
/// @tparam Range Mask range type.
/// @param unit_vectors The input @ref tf::unit_vectors.
/// @param mask Boolean range indicating which unit vectors to keep.
/// @param tag Pass @ref tf::return_index_map to get the mapping.
/// @return Pair of (@ref tf::unit_vectors_buffer, @ref tf::index_map_buffer).
template <typename Index = int, typename Policy, typename Range>
auto reindexed_by_mask(const tf::unit_vectors<Policy> &unit_vectors,
                       const Range &mask, tf::return_index_map_t) {
  auto im = tf::mask_to_index_map<Index>(mask);
  auto out = tf::reindexed(unit_vectors, im);
  return std::make_pair(std::move(out), std::move(im));
}

/// @ingroup reindex
/// @brief Filter unit vectors by boolean mask.
/// @overload
template <typename Index = int, typename Policy, typename Range>
auto reindexed_by_mask(const tf::unit_vectors<Policy> &unit_vectors,
                       const Range &mask) {
  auto im = tf::mask_to_index_map<Index>(mask);
  return tf::reindexed(unit_vectors, im);
}

// =============================================================================
// polygons - deduces from faces()[0][0]
// =============================================================================

/// @ingroup reindex
/// @brief Filter polygons by face mask and return index maps.
///
/// Keeps faces where mask is true. Automatically removes unreferenced points.
///
/// @tparam Index The index type (auto-deduced from geometry if not specified).
/// @tparam Policy The policy type of the polygons.
/// @tparam Range Mask range type.
/// @param polygons The input @ref tf::polygons.
/// @param mask Boolean range indicating which faces to keep.
/// @param tag Pass @ref tf::return_index_map to get the mappings.
/// @return Tuple of (@ref tf::polygons_buffer, face @ref tf::index_map_buffer, point @ref tf::index_map_buffer).
template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_mask(const tf::polygons<Policy> &polygons, const Range &mask,
                       tf::return_index_map_t) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return reindexed_by_mask<ActualIndex>(polygons, mask, tf::return_index_map);
  } else {
    auto face_im = tf::mask_to_index_map<Index>(mask);
    tf::buffer<bool> point_mask;
    point_mask.allocate(polygons.points().size());
    tf::parallel_fill(point_mask, false);
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
/// @brief Filter polygons by face mask.
/// @overload
template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_mask(const tf::polygons<Policy> &polygons,
                       const Range &mask) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return reindexed_by_mask<ActualIndex>(polygons, mask);
  } else {
    return std::get<0>(
        reindexed_by_mask<Index>(polygons, mask, tf::return_index_map));
  }
}

// =============================================================================
// segments - deduces from edges()[0][0]
// =============================================================================

/// @ingroup reindex
/// @brief Filter segments by edge mask and return index maps.
///
/// Keeps edges where mask is true. Automatically removes unreferenced points.
///
/// @tparam Index The index type (auto-deduced from geometry if not specified).
/// @tparam Policy The policy type of the segments.
/// @tparam Range Mask range type.
/// @param segments The input @ref tf::segments.
/// @param mask Boolean range indicating which edges to keep.
/// @param tag Pass @ref tf::return_index_map to get the mappings.
/// @return Tuple of (@ref tf::segments_buffer, edge @ref tf::index_map_buffer, point @ref tf::index_map_buffer).
template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_mask(const tf::segments<Policy> &segments, const Range &mask,
                       tf::return_index_map_t) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return reindexed_by_mask<ActualIndex>(segments, mask, tf::return_index_map);
  } else {
    auto edge_im = tf::mask_to_index_map<Index>(mask);
    tf::buffer<bool> point_mask;
    point_mask.allocate(segments.points().size());
    tf::parallel_fill(point_mask, false);
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
/// @brief Filter segments by edge mask.
/// @overload
template <typename Index = tf::none_t, typename Policy, typename Range>
auto reindexed_by_mask(const tf::segments<Policy> &segments,
                       const Range &mask) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return reindexed_by_mask<ActualIndex>(segments, mask);
  } else {
    return std::get<0>(
        reindexed_by_mask<Index>(segments, mask, tf::return_index_map));
  }
}

} // namespace tf

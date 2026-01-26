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
#include "../core/is_soup.hpp"
#include "../core/none.hpp"
#include "../reindex/return_index_map.hpp"
#include "../reindex/segments.hpp"
#include "./index_map/segments.hpp"
#include "./soup/segments.hpp"

namespace tf {

/// @ingroup clean
/// @brief Remove duplicate/degenerate segments with tolerance.
///
/// Cleans @ref tf::segments by removing duplicate edges, zero-length edges,
/// and points not referenced by any edge. Index type is auto-deduced from
/// segments unless specified. For soups, returns indexed geometry.
///
/// @tparam Index The index type (auto-deduced if not specified).
/// @tparam Policy The policy type of the segments.
/// @param segments The input @ref tf::segments.
/// @param tolerance Points within this distance are considered duplicates.
/// @return A @ref tf::segments_buffer with cleaned geometry.
///
/// @see tf::make_clean_index_map for low-level index map generation.
template <typename Index = tf::none_t, typename Policy>
auto cleaned(const tf::segments<Policy> &segments,
             tf::coordinate_type<Policy> tolerance) {
  if constexpr (std::is_same_v<Index, tf::none_t> && tf::is_soup<Policy>) {
    return cleaned<int>(segments, tolerance);
  } else if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return cleaned<ActualIndex>(segments, tolerance);
  } else if constexpr (tf::is_soup<Policy>) {
    tf::clean::segment_soup<Index, tf::coordinate_type<Policy>,
                            tf::coordinate_dims_v<Policy>>
        out;
    out.build(segments, tolerance);
    return out;
  } else {
    auto [edge_im, point_im] =
        tf::make_clean_index_map<Index>(segments, tolerance);
    return tf::reindexed(segments, edge_im, point_im);
  }
}

/// @ingroup clean
/// @brief Remove exact duplicate/degenerate segments.
/// @overload
template <typename Index = tf::none_t, typename Policy>
auto cleaned(const tf::segments<Policy> &segments) {
  if constexpr (std::is_same_v<Index, tf::none_t> && tf::is_soup<Policy>) {
    return cleaned<int>(segments);
  } else if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(segments.edges()[0][0])>;
    return cleaned<ActualIndex>(segments);
  } else if constexpr (tf::is_soup<Policy>) {
    tf::clean::segment_soup<Index, tf::coordinate_type<Policy>,
                            tf::coordinate_dims_v<Policy>>
        out;
    out.build(segments);
    return out;
  } else {
    auto [edge_im, point_im] = tf::make_clean_index_map<Index>(segments);
    return tf::reindexed(segments, edge_im, point_im);
  }
}

/// @ingroup clean
/// @brief Remove duplicate/degenerate segments with tolerance and return index maps.
/// @overload
///
/// @return Tuple of (@ref tf::segments_buffer, edge @ref tf::index_map_buffer, point @ref tf::index_map_buffer).
template <typename Index = tf::none_t, typename Policy>
auto cleaned(const tf::segments<Policy> &segments,
             tf::coordinate_type<Policy> tolerance, tf::return_index_map_t) {
  static_assert(!tf::is_soup<Policy>, "Soups cannot return index maps.");
  using ActualIndex =
      std::conditional_t<std::is_same_v<Index, tf::none_t>,
                         std::decay_t<decltype(segments.edges()[0][0])>, Index>;
  auto [edge_im, point_im] =
      tf::make_clean_index_map<ActualIndex>(segments, tolerance);
  auto out =
      tf::reindexed(tf::make_segments(segments.edges(), segments.points()),
                    edge_im, point_im);
  return std::make_tuple(std::move(out), std::move(edge_im),
                         std::move(point_im));
}

/// @ingroup clean
/// @brief Remove exact duplicate/degenerate segments and return index maps.
/// @overload
template <typename Index = tf::none_t, typename Policy>
auto cleaned(const tf::segments<Policy> &segments, tf::return_index_map_t) {
  static_assert(!tf::is_soup<Policy>, "Soups cannot return index maps.");
  using ActualIndex =
      std::conditional_t<std::is_same_v<Index, tf::none_t>,
                         std::decay_t<decltype(segments.edges()[0][0])>, Index>;
  auto [edge_im, point_im] = tf::make_clean_index_map<ActualIndex>(segments);
  auto out =
      tf::reindexed(tf::make_segments(segments.edges(), segments.points()),
                    edge_im, point_im);
  return std::make_tuple(std::move(out), std::move(edge_im),
                         std::move(point_im));
}
} // namespace tf

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
#include "../reindex/polygons.hpp"
#include "../reindex/return_index_map.hpp"
#include "./index_map/polygons.hpp"
#include "./soup/polygons.hpp"

namespace tf {

/// @ingroup clean
/// @brief Remove duplicate/degenerate polygons with tolerance.
///
/// Cleans @ref tf::polygons by removing duplicate vertices within tolerance,
/// degenerate faces (less than 3 unique vertices), and unreferenced points.
/// Index type is auto-deduced unless specified. For soups, returns indexed
/// geometry with shared vertices.
///
/// @tparam Index The index type (auto-deduced if not specified).
/// @tparam Policy The policy type of the polygons.
/// @param polygons The input @ref tf::polygons.
/// @param tolerance Points within this distance are considered duplicates.
/// @return A @ref tf::polygons_buffer with cleaned geometry.
///
/// @see tf::make_clean_index_map for low-level index map generation.
template <typename Index = tf::none_t, typename Policy>
auto cleaned(const tf::polygons<Policy> &polygons,
             tf::coordinate_type<Policy> tolerance) {
  if constexpr (std::is_same_v<Index, tf::none_t> && tf::is_soup<Policy>) {
    return cleaned<int>(polygons, tolerance);
  } else if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return cleaned<ActualIndex>(polygons, tolerance);
  } else if constexpr (tf::is_soup<Policy>) {
    tf::clean::polygon_soup<Index, tf::coordinate_type<Policy>,
                            tf::coordinate_dims_v<Policy>,
                            tf::static_size_v<decltype(polygons[0])>>
        out;
    out.build_and_deduplicate_faces(polygons, tolerance);
    return out;
  } else {
    auto [face_im, point_im] =
        tf::make_clean_index_map<Index>(polygons, tolerance);
    return tf::reindexed(tf::make_polygons(polygons.faces(), polygons.points()),
                         face_im, point_im);
  }
}

/// @ingroup clean
/// @brief Remove exact duplicate/degenerate polygons.
/// @overload
template <typename Index = tf::none_t, typename Policy>
auto cleaned(const tf::polygons<Policy> &polygons) {
  if constexpr (std::is_same_v<Index, tf::none_t> && tf::is_soup<Policy>) {
    return cleaned<int>(polygons);
  } else if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return cleaned<ActualIndex>(polygons);
  } else if constexpr (tf::is_soup<Policy>) {
    tf::clean::polygon_soup<Index, tf::coordinate_type<Policy>,
                            tf::coordinate_dims_v<Policy>,
                            tf::static_size_v<decltype(polygons[0])>>
        out;
    out.build_and_deduplicate_faces(polygons);
    return out;
  } else {
    auto [face_im, point_im] = tf::make_clean_index_map<Index>(polygons);
    return tf::reindexed(tf::make_polygons(polygons.faces(), polygons.points()),
                         face_im, point_im);
  }
}

/// @ingroup clean
/// @brief Remove duplicate/degenerate polygons with tolerance and return index maps.
/// @overload
///
/// @return Tuple of (@ref tf::polygons_buffer, face @ref tf::index_map_buffer, point @ref tf::index_map_buffer).
template <typename Index = tf::none_t, typename Policy>
auto cleaned(const tf::polygons<Policy> &polygons,
             tf::coordinate_type<Policy> tolerance, tf::return_index_map_t) {
  static_assert(!tf::is_soup<Policy>, "Soups cannot return index maps.");
  using ActualIndex =
      std::conditional_t<std::is_same_v<Index, tf::none_t>,
                         std::decay_t<decltype(polygons.faces()[0][0])>, Index>;
  auto [face_im, point_im] =
      tf::make_clean_index_map<ActualIndex>(polygons, tolerance);
  auto out =
      tf::reindexed(tf::make_polygons(polygons.faces(), polygons.points()),
                    face_im, point_im);
  return std::make_tuple(std::move(out), std::move(face_im),
                         std::move(point_im));
}

/// @ingroup clean
/// @brief Remove exact duplicate/degenerate polygons and return index maps.
/// @overload
template <typename Index = tf::none_t, typename Policy>
auto cleaned(const tf::polygons<Policy> &polygons, tf::return_index_map_t) {
  static_assert(!tf::is_soup<Policy>, "Soups cannot return index maps.");
  using ActualIndex =
      std::conditional_t<std::is_same_v<Index, tf::none_t>,
                         std::decay_t<decltype(polygons.faces()[0][0])>, Index>;
  auto [face_im, point_im] = tf::make_clean_index_map<ActualIndex>(polygons);
  auto out =
      tf::reindexed(tf::make_polygons(polygons.faces(), polygons.points()),
                    face_im, point_im);
  return std::make_tuple(std::move(out), std::move(face_im),
                         std::move(point_im));
}
} // namespace tf

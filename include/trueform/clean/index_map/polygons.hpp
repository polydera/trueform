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
#include "../../core/algorithm/mask_to_index_map.hpp"
#include "../../core/algorithm/update_by_mask.hpp"
#include "../../core/base/polygons.hpp"
#include "../../core/index_map.hpp"
#include "../../core/none.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../topology/compute_unique_faces_mask.hpp"
#include "../../topology/face_membership.hpp"
#include "./points.hpp"

namespace tf {

namespace clean {
/// @cond INTERNAL
template <typename Range0, typename Range1, typename Index>
auto make_clean_index_map(const tf::core::polygons<Range0, Range1> &polygons,
                          tf::index_map_buffer<Index> &face_map,
                          tf::index_map_buffer<Index> &point_map) {
  // mark polygons that are guaranteed to have zero area with false
  tf::buffer<bool> kept_polygons;
  kept_polygons.allocate(polygons.size());
  tf::parallel_for(
      tf::zip(polygons.faces(), kept_polygons),
      [&](auto begin, auto end) {
        tf::small_vector<Index, 10> buff;
        for (auto &&[face, keep] :
             tf::make_range(std::move(begin), std::move(end))) {
          buff.clear();
          for (auto e : face)
            buff.push_back(point_map.f()[e]);
          std::sort(buff.begin(), buff.end());
          keep = std::unique(buff.begin(), buff.end()) - buff.begin() > 2;
        }
      },
      tf::checked);

  tf::mask_to_index_map(kept_polygons, face_map);
  // mark points that are contained in any polygon
  auto &contained_points = kept_polygons;
  contained_points.allocate(point_map.kept_ids().size());
  tf::parallel_fill(contained_points, false);
  tf::parallel_for_each(
      tf::make_indirect_range(
          face_map.kept_ids(),
          tf::make_block_indirect_range(polygons.faces(), point_map.f())),
      [&](const auto &face) {
        for (auto e : face)
          contained_points[e] = true;
      },
      tf::checked);
  tf::update_by_mask(point_map, contained_points);

  auto remapped_faces = tf::make_faces(tf::make_indirect_range(
      face_map.kept_ids(),
      tf::make_block_indirect_range(polygons.faces(), point_map.f())));
  kept_polygons.allocate(remapped_faces.size());
  tf::face_membership<Index> fm;
  fm.build(remapped_faces, point_map.kept_ids().size());
  tf::compute_unique_faces_mask(remapped_faces, fm, kept_polygons);
  tf::update_by_mask(face_map, kept_polygons);
}
/// @endcond
} // namespace clean

/// @ingroup clean
/// @brief Generate index maps for polygon cleaning (output parameters).
///
/// Creates index maps for both faces and points.
/// Use @ref tf::reindexed to apply the maps to associated data.
///
/// @tparam Range0 The face range type.
/// @tparam Range1 The point range type.
/// @tparam Index The index type.
/// @param polygons The input @ref tf::polygons.
/// @param face_map Output face @ref tf::index_map_buffer to populate.
/// @param point_map Output point @ref tf::index_map_buffer to populate.
template <typename Range0, typename Range1, typename Index>
auto make_clean_index_map(const tf::core::polygons<Range0, Range1> &polygons,
                          tf::index_map_buffer<Index> &face_map,
                          tf::index_map_buffer<Index> &point_map) {
  if (!polygons.size())
    return;
  make_clean_index_map(polygons.points(), point_map);
  clean::make_clean_index_map(polygons, face_map, point_map);
}

/// @ingroup clean
/// @brief Generate index maps for polygon cleaning with tolerance (output
/// parameters).
/// @overload
template <typename Range0, typename Range1, typename Index>
auto make_clean_index_map(
    const tf::core::polygons<Range0, Range1> &polygons,
    tf::coordinate_type<decltype(polygons.points())> tolerance,
    tf::index_map_buffer<Index> &face_map,
    tf::index_map_buffer<Index> &point_map) {
  if (!polygons.size())
    return;
  make_clean_index_map(polygons.points(), tolerance, point_map);
  clean::make_clean_index_map(polygons, face_map, point_map);
}

/// @ingroup clean
/// @brief Generate index maps for exact polygon deduplication.
///
/// Creates index maps for both faces and points.
/// Use @ref tf::reindexed to apply the maps to associated data.
///
/// @tparam Index The index type (auto-deduced if not specified).
/// @tparam Range0 The face range type.
/// @tparam Range1 The point range type.
/// @param polygons The input @ref tf::polygons.
/// @return Tuple of (face @ref tf::index_map_buffer, point @ref
/// tf::index_map_buffer).
template <typename Index = tf::none_t, typename Range0, typename Range1>
auto make_clean_index_map(const tf::core::polygons<Range0, Range1> &polygons) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return make_clean_index_map<ActualIndex>(polygons);
  } else {
    tf::index_map_buffer<Index> face_map;
    tf::index_map_buffer<Index> point_map;
    make_clean_index_map(polygons, face_map, point_map);
    return std::make_pair(std::move(face_map), std::move(point_map));
  }
}

/// @ingroup clean
/// @brief Generate index maps for polygon cleaning with tolerance.
/// @overload
template <typename Index = tf::none_t, typename Range0, typename Range1>
auto make_clean_index_map(const tf::core::polygons<Range0, Range1> &polygons,
                          tf::coordinate_type<Range1> tolerance) {
  if constexpr (std::is_same_v<Index, tf::none_t>) {
    using ActualIndex = std::decay_t<decltype(polygons.faces()[0][0])>;
    return make_clean_index_map<ActualIndex>(polygons, tolerance);
  } else {
    tf::index_map_buffer<Index> face_map;
    tf::index_map_buffer<Index> point_map;
    make_clean_index_map(polygons, tolerance, face_map, point_map);
    return std::make_pair(std::move(face_map), std::move(point_map));
  }
}
} // namespace tf

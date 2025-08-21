/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../../core/algorithm/mask_to_index_map.hpp"
#include "../../core/algorithm/update_by_mask.hpp"
#include "../../core/base/polygons.hpp"
#include "../../core/index_map.hpp"
#include "../../core/small_vector.hpp"
#include "./points.hpp"

namespace tf {
namespace clean {
template <typename Range0, typename Range1, typename Index>
auto make_clean_index_map(const tf::core::polygons<Range0, Range1> &polygons,
                          tf::index_map_buffer<Index> &face_map,
                          tf::index_map_buffer<Index> &point_map) {
  // mark polygons that are guaranteed to have zero area with false
  tf::buffer<bool> kept_polygons;
  kept_polygons.allocate(polygons.size());
  tf::parallel_for(tf::zip(polygons.faces(), kept_polygons), [&](auto r) {
    tf::small_vector<Index, 10> buff;
    for (auto &&[face, keep] : r) {
      buff.clear();
      for (auto e : face)
        buff.push_back(point_map.f()[e]);
      std::sort(buff.begin(), buff.end());
      keep = std::unique(face.begin(), face.end()) - face.begin() > 2;
    }
  });

  tf::mask_to_index_map(kept_polygons, face_map);
  // mark points that are contained in any polygon
  auto &contained_points = kept_polygons;
  contained_points.allocate(polygons.points().size());
  tf::parallel_fill(contained_points, false);
  tf::parallel_apply(
      tf::make_indirect_range(face_map.kept_ids(), polygons.faces()),
      [&](const auto &face) {
        for (auto e : face)
          contained_points[e] = true;
      },
      tf::checked);
  tf::update_by_mask(point_map, contained_points);
}
} // namespace clean

template <typename Range0, typename Range1, typename Index>
auto make_clean_index_map(const tf::core::polygons<Range0, Range1> &polygons,
                          tf::index_map_buffer<Index> &edge_map,
                          tf::index_map_buffer<Index> &point_map) {
  if(!polygons.size())
    return;
  make_clean_index_map(polygons.points(), point_map);
  clean::make_clean_index_map(polygons, edge_map, point_map);
}

template <typename Range0, typename Range1, typename RealType, typename Index>
auto make_clean_index_map(
    const tf::core::polygons<Range0, Range1> &polygons,
    tf::coordinate_type<decltype(polygons.points())> tolerance,
    tf::index_map_buffer<Index> &edge_map,
    tf::index_map_buffer<Index> &point_map) {
  if(!polygons.size())
    return;
  make_clean_index_map(polygons.points(), tolerance, point_map);
  clean::make_clean_index_map(polygons, edge_map, point_map);
}
} // namespace tf

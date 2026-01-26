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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/concatenated_blocked_ranges.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../reindex/concatenated.hpp"
#include "../loop/vertex_source.hpp"
#include "./polygon_arrangement_ids.hpp"
#include "./triangulate_cut_faces.hpp"

namespace tf::cut {

template <typename Policy, typename Policy1, typename Index, typename Range0,
          typename Range1>
auto make_boolean_common(const tf::polygons<Policy> _polygons,
                         const tf::points<Policy1> &intersection_points,
                         tf::cut::polygon_arrangement_ids<Index> &pai,
                         const Range0 &descriptors, const Range1 &mapped_loops,
                         std::size_t index, tf::direction direction) {
  auto make_polygons = [](const auto &form) {
    return tf::wrap_map(form, [](auto &&x) {
      return tf::core::make_polygons(
          x.faces(), x.points().template as<tf::coordinate_type<Policy1>>());
    });
  };
  auto polygons = make_polygons(_polygons);

  tf::buffer<Index> original_map;
  original_map.allocate(polygons.points().size());
  tf::parallel_fill(original_map, -1);
  tf::buffer<Index> created_map;
  created_map.allocate(intersection_points.size());
  tf::parallel_fill(created_map, -1);
  Index original_current = 0;
  Index create_current = 0;
  tf::buffer<Index> original_ids;
  original_ids.reserve(polygons.points().size());
  tf::buffer<Index> created_ids;
  created_ids.reserve(created_map.size());
  for (const auto &loop :
       tf::make_indirect_range(pai.cut_faces[index], mapped_loops))
    for (auto v : loop) {
      if (v.source == tf::loop::vertex_source::created) {
        if (created_map[v.id] == -1) {
          created_ids.push_back(v.id);
          created_map[v.id] = create_current++;
        }
      } else {
        if (original_map[v.id] == -1) {
          original_map[v.id] = original_current++;
          original_ids.push_back(v.id);
        }
      }
    }

  for (const auto &face :
       tf::make_indirect_range(pai.polygons[index], polygons.faces())) {
    for (auto v : face)
      if (original_map[v] == -1) {
        original_map[v] = original_current++;
        original_ids.push_back(v);
      }
  }

  auto map_vertex_f = [&](auto v) {
    if (v.source == tf::loop::vertex_source::created)
      return created_map[v.id] + original_current;
    else
      return original_map[v.id];
  };

  auto make_projector = [&](auto d) {
    auto frame = tf::frame_of(polygons);
    auto proj = tf::make_simple_projector(
        tf::transformed_normal(tf::make_normal(polygons[d.object]), frame));
    return [proj, &polygons, &intersection_points, frame](auto v) {
      if (v.source == tf::loop::vertex_source::original)
        return proj(tf::transformed(polygons.points()[v.id], frame));
      else
        return proj(intersection_points[v.id]);
    };
  };

  tf::blocked_buffer<Index, 3> triangles;
  tf::cut::triangulate_cut_faces(
      tf::make_indirect_range(pai.cut_faces[index],
                              tf::zip(descriptors, mapped_loops)),
      make_projector, map_vertex_f, triangles.data_buffer());

  auto mapped_faces = tf::make_indirect_range(
      pai.polygons[index],
      tf::make_block_indirect_range(polygons.faces(), original_map));

  auto faces = tf::core::concatenated_blocked_ranges_directed<Index>(
      std::make_pair(tf::make_range(mapped_faces), direction),
      std::make_pair(tf::make_range(triangles), direction));
  auto make_mapped_points = [](const auto &ids, const auto &polygons) {
    auto frame = tf::frame_of(polygons);
    return tf::make_points(tf::make_indirect_range(
        ids, tf::make_mapped_range(polygons.points(), [frame](auto pt) {
          return tf::transformed(pt, frame);
        })));
  };
  auto points = tf::concatenated(
      make_mapped_points(original_ids, _polygons),
      tf::make_points(tf::make_indirect_range(created_ids, intersection_points))
          .template as<tf::coordinate_type<Policy>>());
  return tf::make_polygons_buffer(std::move(faces), std::move(points));
}
} // namespace tf::cut

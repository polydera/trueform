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
#include "../../core/concatenated_blocked_range_collections.hpp"
#include "../../core/polygons.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/take.hpp"
#include "../../intersect/types/simple_intersections.hpp"
#include "../scalar_cut_faces.hpp"
#include "./scalars.hpp"
#include "./triangulate_cut_faces.hpp"

namespace tf::cut {
template <typename LabelType, typename Policy, typename Index, typename RealT,
          std::size_t Dims, typename Range, typename Iterator0, std::size_t N0,
          typename Iterator1, std::size_t N1>
auto make_isobands(
    const tf::polygons<Policy> &polygons,
    const tf::intersect::simple_intersections<Index, RealT, Dims> &sfi,
    const tf::scalar_cut_faces<Index> &scf, const Range &scalars,
    tf::range<Iterator0, N0> cut_values,
    tf::range<Iterator1, N1> selected_bands) {
  auto all_ids_ = tf::cut::make_polygon_arrangement_ids<LabelType>(
      polygons, sfi, scf, scalars, cut_values);

  auto polygon_ids = tf::make_indirect_range(selected_bands, all_ids_.polygons);
  auto cut_ids = tf::make_indirect_range(selected_bands, all_ids_.cut_faces);
  std::size_t polygon_size = 0;
  for (const auto &r : polygon_ids)
    polygon_size += r.size();

  tf::buffer<Index> created_map;
  created_map.allocate(sfi.intersection_points().size());
  tf::parallel_fill(created_map, -1);
  tf::buffer<Index> original_map;
  original_map.allocate(polygons.points().size());
  tf::parallel_fill(original_map, -1);
  Index original_current = 0;
  Index created_current = 0;
  tf::buffer<Index> created_ids;
  created_ids.reserve(sfi.intersection_points().size());
  tf::buffer<Index> original_ids;
  original_ids.reserve(polygons.points().size());
  for (auto &&loops :
       tf::make_block_indirect_range(cut_ids, scf.mapped_loops())) {
    for (auto loop : loops) {
      for (auto v : loop) {
        if (v.source == tf::loop::vertex_source::created) {
          if (created_map[v.id] == -1) {
            created_map[v.id] = created_current++;
            created_ids.push_back(v.id);
          }
        } else {
          if (original_map[v.id] == -1) {
            original_map[v.id] = original_current++;
            original_ids.push_back(v.id);
          }
        }
      }
    }
  }
  for (auto faces :
       tf::make_block_indirect_range(polygon_ids, polygons.faces()))
    for (const auto &face : faces)
      for (auto id : face)
        if (original_map[id] == -1) {
          original_map[id] = original_current++;
          original_ids.push_back(id);
        }

  auto map_vertex_f = [&](auto v) {
    if (v.source == tf::loop::vertex_source::created)
      return created_map[v.id] + original_current;
    else
      return original_map[v.id];
  };

  auto make_projector = [&](auto d, const auto &polygons) {
    auto proj = tf::make_simple_projector(tf::make_normal(polygons[d.object]));
    return [proj, &polygons, &sfi](auto v) {
      if (v.source == tf::loop::vertex_source::original)
        return proj(polygons.points()[v.id]);
      else
        return proj(sfi.intersection_points()[v.id]);
    };
  };

  tf::points_buffer<tf::coordinate_type<Policy>, tf::coordinate_dims_v<Policy>>
      points_out;
  points_out.allocate(created_ids.size() + original_ids.size());
  tf::parallel_copy(tf::make_indirect_range(original_ids, polygons.points()),
                    tf::take(points_out, original_ids.size()));
  tf::parallel_copy(
      tf::make_indirect_range(created_ids, sfi.intersection_points()),
      tf::drop(points_out, original_ids.size()));
  tf::buffer<Index> cf_offsets;
  tf::blocked_buffer<Index, 3> triangles;

  tf::generate_offset_blocks(
      cut_ids, cf_offsets, triangles, [&](const auto &ids, auto &data) {
        tf::cut::triangulate_cut_faces(
            tf::make_indirect_range(
                ids, tf::zip(scf.descriptors(), scf.mapped_loops())),
            [&](auto d) { return make_projector(d, polygons); }, map_vertex_f,
            data.data_buffer());
      });

  auto mapped_faces =
      tf::make_block_indirect_range(polygons.faces(), original_map);

  auto faces = tf::concatenated_blocked_range_collections<Index>(
      tf::make_block_indirect_range(polygon_ids, mapped_faces),
      tf::make_offset_block_range(cf_offsets, triangles));
  tf::buffer<LabelType> labels;
  labels.allocate(faces.size());
  auto cut_labels = tf::drop(labels, polygon_size);
  auto original_labels = tf::take(labels, polygon_size);
  tf::parallel_for_each(
      tf::zip(selected_bands,
              tf::make_offset_block_range(cf_offsets, cut_labels)),
      [](auto pair) {
        auto [id, r] = pair;
        std::fill(r.begin(), r.end(), id);
      },
      tf::checked);
  Index start = 0;
  for (const auto &[id, r] : tf::zip(selected_bands, polygon_ids)) {
    Index end = start + r.size();
    tf::parallel_fill(tf::slice(original_labels, start, end), id);
    start = end;
  }
  return std::make_tuple(
      tf::make_polygons_buffer(std::move(faces), std::move(points_out)),
      std::move(labels), std::move(created_ids));
}
} // namespace tf::cut

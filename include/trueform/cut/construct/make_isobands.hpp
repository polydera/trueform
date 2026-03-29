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

#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/concatenated_blocked_range_collections.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/take.hpp"
#include "../../core/views/zip.hpp"
#include "../../exact/projection_axes.hpp"
#include "../../exact/pt_converter.hpp"
#include "../../intersect/graph/vertex.hpp"
#include "../../intersect/types/simple_intersections.hpp"
#include "../face_cuts.hpp"
#include "../partition/partition_ids.hpp"
#include "./triangulate_arrangement_cuts.hpp"

namespace tf::cut {

template <typename LabelType, typename Index, typename Policy, typename RealT,
          std::size_t Dims, typename SelectedBands>
auto make_isobands(
    const tf::polygons<Policy> &polygons,
    const tf::intersect::simple_intersections<Index, RealT, Dims> &si,
    const tf::face_cuts<Index> &fc,
    const tf::cut::partition_ids<Index> &pids,
    const SelectedBands &selected_bands) {
  using vertex_t = tf::intersect::graph::vertex<Index>;
  using source = tf::intersect::graph::vertex_source;

  auto polygon_ids =
      tf::make_indirect_range(selected_bands, pids.polygons);
  auto cut_ids =
      tf::make_indirect_range(selected_bands, pids.cut_faces);

  auto loops = fc.loops();
  auto descs = fc.descriptors();

  // Vertex remapping: only vertices referenced by selected bands
  tf::buffer<Index> original_map;
  original_map.allocate(polygons.points().size());
  const Index sentinel_orig = static_cast<Index>(original_map.size());
  tf::parallel_fill(original_map, sentinel_orig);

  tf::buffer<Index> created_map;
  created_map.allocate(si.intersection_points().size());
  const Index sentinel_created = static_cast<Index>(created_map.size());
  tf::parallel_fill(created_map, sentinel_created);

  tf::buffer<Index> original_ids;
  original_ids.reserve(polygons.points().size());
  tf::buffer<Index> created_ids;
  created_ids.reserve(si.intersection_points().size());

  Index original_current = 0;
  Index created_current = 0;

  // Pass 1: cut face loops for selected bands
  for (auto &&band_loops :
       tf::make_block_indirect_range(cut_ids, loops)) {
    for (auto loop : band_loops) {
      for (const auto &v : loop) {
        if (v.source == source::created) {
          if (created_map[v.id] == sentinel_created) {
            created_map[v.id] = created_current++;
            created_ids.push_back(v.id);
          }
        } else {
          if (original_map[v.id] == sentinel_orig) {
            original_map[v.id] = original_current++;
            original_ids.push_back(v.id);
          }
        }
      }
    }
  }

  // Pass 2: uncut polygons for selected bands
  for (auto faces :
       tf::make_block_indirect_range(polygon_ids, polygons.faces()))
    for (const auto &face : faces)
      for (auto v : face)
        if (original_map[v] == sentinel_orig) {
          original_map[v] = original_current++;
          original_ids.push_back(v);
        }

  auto map_vertex = [&](auto, const vertex_t &v) -> Index {
    if (v.source == source::original)
      return original_map[v.id];
    return created_map[v.id] + original_current;
  };

  auto conv = tf::exact::make_pt_converter(polygons);
  auto ipts = si.intersection_points();

  auto make_projector = [&](const auto &desc) {
    auto fp = polygons[desc.object];
    auto axes = tf::exact::projection_axes(conv(fp[0]), conv(fp[1]), conv(fp[2]));
    return [axes, &conv, &polygons, ipts](const vertex_t &v)
               -> tf::point<int32_t, 2> {
      auto pt = (v.source == source::original) ? conv(polygons.points()[v.id])
                                               : conv(ipts[v.id]);
      return {pt[axes.first], pt[axes.second]};
    };
  };

  tf::buffer<Index> cf_offsets;
  tf::blocked_buffer<Index, 3> triangles;

  tf::generate_offset_blocks(
      cut_ids, cf_offsets, triangles, [&](const auto &ids, auto &tri_buf) {
        tf::cut::triangulate_partition_cuts<Index>(
            tf::make_indirect_range(ids, tf::zip(descs, loops)),
            make_projector, map_vertex, tri_buf.data_buffer());
      });

  // Build uncut face ranges (remapped)
  auto mapped_faces =
      tf::make_block_indirect_range(polygons.faces(), original_map);

  auto faces = tf::concatenated_blocked_range_collections<Index>(
      tf::make_block_indirect_range(polygon_ids, mapped_faces),
      tf::make_offset_block_range(cf_offsets, triangles));

  // Build points
  tf::points_buffer<tf::coordinate_type<Policy>, tf::coordinate_dims_v<Policy>>
      pts_out;
  pts_out.allocate(original_current + created_current);
  tf::parallel_copy(
      tf::make_indirect_range(original_ids, polygons.points()),
      tf::take(pts_out, original_current));
  tf::parallel_copy(
      tf::make_indirect_range(created_ids, si.intersection_points()),
      tf::drop(pts_out, original_current));

  // Build labels
  std::size_t polygon_size = 0;
  for (auto r : polygon_ids)
    polygon_size += r.size();

  tf::buffer<LabelType> labels;
  labels.allocate(faces.size());
  auto original_labels = tf::take(labels, polygon_size);
  auto cut_labels = tf::drop(labels, polygon_size);

  tf::parallel_for_each(
      tf::zip(selected_bands,
              tf::make_offset_block_range(cf_offsets, cut_labels)),
      [](auto pair) {
        auto [id, r] = pair;
        std::fill(r.begin(), r.end(), id);
      },
      tf::checked);

  Index start = 0;
  for (auto [id, r] : tf::zip(selected_bands, polygon_ids)) {
    Index end = start + r.size();
    tf::parallel_fill(tf::slice(original_labels, start, end), id);
    start = end;
  }

  return std::make_tuple(
      tf::make_polygons_buffer(std::move(faces), std::move(pts_out)),
      std::move(labels), std::move(created_ids));
}

} // namespace tf::cut

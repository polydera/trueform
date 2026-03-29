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
#include "../../core/concatenated_blocked_ranges.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/enumerate.hpp"
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
          std::size_t Dims>
auto embedded_isocurves(
    const tf::polygons<Policy> &polygons,
    const tf::intersect::simple_intersections<Index, RealT, Dims> &si,
    const tf::face_cuts<Index> &fc,
    const tf::cut::partition_ids<Index> &pids) {
  using vertex_t = tf::intersect::graph::vertex<Index>;
  using source = tf::intersect::graph::vertex_source;

  const Index n_orig = static_cast<Index>(polygons.points().size());
  const Index n_ip = static_cast<Index>(si.intersection_points().size());

  tf::points_buffer<tf::coordinate_type<Policy>, tf::coordinate_dims_v<Policy>>
      pts_buf;
  pts_buf.allocate(n_orig + n_ip);
  tf::parallel_copy(polygons.points(), tf::take(pts_buf, n_orig));
  tf::parallel_copy(si.intersection_points(), tf::drop(pts_buf, n_orig));

  auto map_vertex = [&](auto, const vertex_t &v) -> Index {
    return (v.source == source::created) ? (n_orig + v.id) : v.id;
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

  auto loops = fc.loops();
  auto descs = fc.descriptors();

  tf::buffer<Index> cf_offsets;
  tf::blocked_buffer<Index, 3> triangles;

  tf::generate_offset_blocks(
      pids.cut_faces, cf_offsets, triangles,
      [&](const auto &ids, auto &tri_buf) {
        tf::cut::triangulate_partition_cuts<Index>(
            tf::make_indirect_range(ids, tf::zip(descs, loops)),
            make_projector, map_vertex, tri_buf.data_buffer());
      });

  auto faces = tf::concatenated_blocked_ranges<Index>(
      tf::make_indirect_range(pids.polygons.data_buffer(), polygons.faces()),
      triangles);

  tf::buffer<LabelType> labels;
  labels.allocate(faces.size());
  std::size_t polygon_size = pids.polygons.data_buffer().size();
  auto original_labels = tf::take(labels, polygon_size);
  auto cut_labels = tf::drop(labels, polygon_size);

  tf::parallel_for_each(
      tf::enumerate(tf::make_offset_block_range(cf_offsets, cut_labels)),
      [](auto pair) {
        auto [id, r] = pair;
        std::fill(r.begin(), r.end(), id);
      },
      tf::checked);

  Index start = 0;
  for (const auto &[id, r] : tf::enumerate(pids.polygons)) {
    Index end = start + r.size();
    tf::parallel_fill(tf::slice(original_labels, start, end), id);
    start = end;
  }

  return std::make_pair(
      tf::make_polygons_buffer(std::move(faces), std::move(pts_buf)),
      std::move(labels));
}

} // namespace tf::cut

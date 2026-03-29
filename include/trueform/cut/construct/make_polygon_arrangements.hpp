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
#include "../../core/concatenated_blocked_ranges.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../exact/projection_axes.hpp"
#include "../../exact/vertex_converter.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
#include "../face_cuts.hpp"
#include "./make_arrangement_map_data.hpp"
#include "./triangulate_arrangement_cuts.hpp"

namespace tf::cut {

template <typename Index, typename Policy, typename RealType>
auto make_polygon_arrangements(
    const tf::polygons<Policy> &polygons,
    const tf::intersection_graph<Index> &ig, const tf::face_cuts<Index> &fc,
    const tf::exact::vertex_converter<RealType, 3> &converter) {
  auto ipts = ig.points();
  auto map_data = tf::cut::make_embed_map_data<Index>(fc, polygons, Index(0));

  auto descs_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.descriptors());
  auto loops_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.loops());

  auto make_projector = [&](const auto &desc) {
    auto object = desc.object;
    auto face = polygons.faces()[object];
    auto get_pt = [&](Index vid) -> tf::point<int32_t, 3> {
      return converter.convert(
          tf::transformed(polygons.points()[vid], tf::frame_of(polygons)));
    };
    auto axes = tf::exact::projection_axes(get_pt(face[0]), get_pt(face[1]),
                                           get_pt(face[2]));
    return [axes, &converter, ipts,
            &polygons](const auto &v) -> tf::point<int32_t, 2> {
      tf::point<int32_t, 3> pt;
      if (v.source == tf::intersect::graph::vertex_source::original)
        pt = converter.convert(
            tf::transformed(polygons.points()[v.id], tf::frame_of(polygons)));
      else
        pt = ipts[v.id];
      return {pt[axes.first], pt[axes.second]};
    };
  };

  auto map_vertex = [&](auto, const auto &v) {
    return map_data.map_vertex(v);
  };

  tf::buffer<Index> tri_data;
  tf::buffer<Index> tri_origins;
  tf::cut::triangulate_partition_cuts<Index>(
      tf::zip(descs_per_tag[0], loops_per_tag[0]), make_projector, map_vertex,
      tri_data, tri_origins);

  auto triangles = tf::make_blocked_range<3>(tf::make_range(tri_data));

  auto mapped_faces = tf::make_indirect_range(
      map_data.uncut_face_ids,
      tf::make_block_indirect_range(polygons.faces(), map_data.original_map));

  auto faces = tf::core::concatenated_blocked_ranges_directed<Index>(
      std::make_pair(tf::make_range(mapped_faces), tf::direction::forward),
      std::make_pair(tf::make_range(triangles), tf::direction::forward));

  auto n_created = static_cast<Index>(ipts.size());
  tf::points_buffer<RealType, 3> pts_buf;
  pts_buf.allocate(map_data.n_original_points + n_created);

  auto frame = tf::frame_of(polygons);
  tf::parallel_copy(
      tf::make_points(tf::make_indirect_range(
          map_data.original_ids,
          tf::make_mapped_range(
              polygons.points(),
              [frame](auto pt) { return tf::transformed(pt, frame); }))),
      tf::take(pts_buf, map_data.n_original_points));

  tf::parallel_copy(
      tf::make_points(tf::make_mapped_range(
          ipts, [&converter](auto pt) { return converter.deconvert(pt); })),
      tf::drop(pts_buf, map_data.n_original_points));

  // Face labels: uncut → original face ID, triangulated → desc.object
  auto n_uncut = static_cast<Index>(map_data.uncut_face_ids.size());
  tf::buffer<Index> face_labels;
  face_labels.allocate(faces.size());
  tf::parallel_copy(map_data.uncut_face_ids, tf::take(face_labels, n_uncut));
  tf::parallel_copy(tri_origins, tf::drop(face_labels, n_uncut));

  return std::make_tuple(
      tf::make_polygons_buffer(std::move(faces), std::move(pts_buf)),
      std::move(face_labels), std::move(map_data));
}

} // namespace tf::cut

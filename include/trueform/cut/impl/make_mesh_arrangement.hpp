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
#include "../../core/frame_of.hpp"
#include "../../core/normal.hpp"
#include "../../core/projector.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../intersect/exact/intersections_between_polygons.hpp"
#include "../../intersect/exact/vertex_converter.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
#include "../../reindex/concatenated.hpp"
#include "../face_cuts.hpp"
#include "./make_arrangement_map_data.hpp"
#include "./triangulate_arrangement_cuts.hpp"

namespace tf::cut {

/// Build a mesh arrangement from a range of tagged polygon forms.
///
/// Takes the intersection graph, face cuts, and a range of polygon forms.
/// Returns (mesh, tag_labels, face_labels).
template <typename Index, typename FormsRange, typename RealType>
auto make_mesh_arrangement(
    const tf::intersection_graph<Index> &ig, const tf::face_cuts<Index> &fc,
    const FormsRange &forms,
    const tf::exact::vertex_converter<RealType, 3> &converter) {
  auto n_meshes = static_cast<Index>(forms.size());
  auto apply_to_polygons = [&](Index tag, const auto &f) { f(forms[tag]); };

  // 1. Build maps
  auto map_data = tf::cut::make_arrangement_map_data<Index>(
      fc, apply_to_polygons, n_meshes);

  // 2. Triangulate cut faces
  tf::buffer<Index> tri_data;
  tf::buffer<Index> tri_tags;
  tf::buffer<Index> tri_origins;

  auto make_projector = [&](const auto &desc) {
    auto tag = desc.tag;
    auto object = desc.object;
    auto frame = tf::frame_of(forms[tag]);
    auto proj = tf::make_simple_projector(
        tf::transformed_normal(tf::make_normal(forms[tag][object]), frame));
    auto pts = ig.points();
    return [proj, &forms, &converter, pts, frame,
            tag](const auto &v) -> tf::point<double, 2> {
      if (v.source == tf::intersect::graph::vertex_source::original)
        return proj(tf::transformed(forms[tag].points()[v.id], frame));
      return proj(converter.deconvert(pts[v.id]));
    };
  };

  tf::cut::triangulate_arrangement_cuts<Index>(
      tf::zip(fc.descriptors(), fc.loops()), make_projector, map_data, tri_data,
      tri_tags, tri_origins);

  auto triangles = tf::make_blocked_range<3>(tf::make_range(tri_data));

  auto original_maps = tf::make_offset_block_range(map_data.point_offsets,
                                                   map_data.original_map);

  // 3. Per-mesh uncut face ranges (remapped vertex IDs), lazily
  auto uncut_faces =
      tf::make_mapped_range(tf::make_sequence_range(n_meshes), [&](Index t) {
        auto off = map_data.original_offsets[t];
        return tf::make_indirect_range(
            map_data.original_face_ids[t],
            tf::make_block_indirect_range(
                forms[t].faces(),
                tf::make_mapped_range(original_maps[t],
                                      [off](Index x) { return x + off; })));
      });

  // 4. Concatenate faces: uncut (per mesh) + triangulated cuts
  auto faces =
      tf::concatenated_blocked_range_collections<Index>(uncut_faces, tf::make_range(&triangles, 1));

  // 5. Build points (parallel per mesh + intersection points)
  auto total_pts =
      map_data.total_original_points + static_cast<Index>(ig.points().size());
  tf::points_buffer<RealType, 3> pts_buf;
  pts_buf.allocate(total_pts);

  {
    auto pts_range = tf::make_offset_block_range(
        map_data.original_offsets, pts_buf);
    tbb::task_group tg;
    for (Index t = 0; t < n_meshes; ++t) {
      tg.run([&, t] {
        auto frame = tf::frame_of(forms[t]);
        tf::parallel_copy(
            tf::make_points(tf::make_indirect_range(
                map_data.original_ids[t],
                tf::make_mapped_range(forms[t].points(), [frame](auto pt) {
                  return tf::transformed(pt, frame);
                }))),
            pts_range[t]);
      });
    }
    tg.run([&] {
      auto ipts = ig.points();
      tf::parallel_copy(
          tf::make_points(tf::make_mapped_range(
              ipts, [&converter](auto pt) { return converter.deconvert(pt); })),
          tf::drop(pts_buf, map_data.total_original_points));
    });
    tg.wait();
  }

  // 6. Build labels (parallel)
  auto total_faces = static_cast<Index>(faces.size());
  tf::buffer<Index> tag_labels;
  tf::buffer<Index> face_labels;
  tag_labels.allocate(total_faces);
  face_labels.allocate(total_faces);

  {
    auto tag_uncut = tf::make_offset_block_range(
        map_data.original_face_offsets, tag_labels);
    auto face_uncut = tf::make_offset_block_range(
        map_data.original_face_offsets, face_labels);
    tbb::task_group tg;
    for (Index t = 0; t < n_meshes; ++t) {
      tg.run([&, t] {
        tf::parallel_fill(tag_uncut[t], t);
        tf::parallel_copy(map_data.original_face_ids[t], face_uncut[t]);
      });
    }
    tg.run([&] {
      auto tri_off = map_data.total_original_faces;
      tf::parallel_copy(tri_tags, tf::drop(tag_labels, tri_off));
      tf::parallel_copy(tri_origins, tf::drop(face_labels, tri_off));
    });
    tg.wait();
  }

  return std::make_tuple(
      tf::make_polygons_buffer(std::move(faces), std::move(pts_buf)),
      std::move(tag_labels), std::move(face_labels));
}

} // namespace tf::cut

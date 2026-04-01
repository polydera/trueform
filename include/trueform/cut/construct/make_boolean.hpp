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

#include "../../core/algorithm/ids_to_index_map.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/concatenated_blocked_ranges.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/index_map.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/stitch_index_maps.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../exact/projection_axes.hpp"
#include "../../exact/vertex_converter.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
#include "../../reindex/return_index_map.hpp"
#include "../arrangement_class.hpp"
#include "../boolean_config.hpp"
#include "../classification/make_classifications.hpp"
#include "../cut_graph.hpp"
#include "../face_cuts.hpp"
#include "../partition/make_ids.hpp"
#include "./make_arrangement_map_data.hpp"
#include "./triangulate_arrangement_cuts.hpp"
#include "tbb/task_group.h"

namespace tf::cut {

/// Build a boolean result mesh from two intersected meshes.
///
/// Runs the full pipeline: classification → partition → vertex remapping →
/// triangulation → mesh assembly. Returns (mesh, labels) or
/// (mesh, labels, stitch_index_maps) when MakeMaps is true.
template <typename LabelType, typename Index, typename Policy0,
          typename Policy1, typename RealType, typename Int,
          bool MakeMaps = false>
auto make_boolean(
    const tf::polygons<Policy0> &polygons0,
    const tf::polygons<Policy1> &polygons1,
    const tf::intersection_graph<Index, Int> &ig,
    const tf::face_cuts<Index, Int> &fc, const tf::cut_graph<Index> &cg,
    const tf::exact::vertex_converter<Int, RealType, 3> &converter,
    std::array<tf::arrangement_class, 2> classes,
    const tf::boolean_config &config,
    std::integral_constant<bool, MakeMaps> = {}) {
  auto ipts = ig.points();

  // 1. Classification → include/exclude labels
  auto cls_labels = tf::cut::make_classification_labels<LabelType>(
      polygons0, polygons1, fc, cg, converter, tf::make_points(ipts), classes,
      config);

  // 2. Partition IDs — which faces belong to include (label=1)
  tf::small_vector<tf::cut::partition_ids<Index>, 4> pids;
  pids.resize(2);
  tbb::parallel_invoke(
      [&] { pids[0] = tf::cut::make_partition_ids<Index>(cls_labels[0]); },
      [&] { pids[1] = tf::cut::make_partition_ids<Index>(cls_labels[1]); });

  tf::small_vector<LabelType, 4> include_labels = {1, 1};
  auto apply_to_polygons = [&](Index tag, const auto &f) {
    if (tag == 0)
      f(polygons0);
    else
      f(polygons1);
  };

  // 3. Vertex remapping for selected faces
  auto map_data =
      tf::cut::make_partition_map_data(fc, static_cast<Index>(ipts.size()),
                                       pids, include_labels, apply_to_polygons);

  // 4. Triangulate selected cut faces (per mesh, parallel)
  auto descs_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.descriptors());
  auto loops_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.loops());

  auto make_projector = [&](const auto &desc) {
    auto tag = desc.tag;
    auto object = desc.object;
    auto get_pt = [&, tag](Index vid) -> tf::point<Int, 3> {
      if (tag == 0)
        return converter.convert(
            tf::transformed(polygons0.points()[vid], tf::frame_of(polygons0)));
      else
        return converter.convert(
            tf::transformed(polygons1.points()[vid], tf::frame_of(polygons1)));
    };
    auto make_axes = [&](auto face) {
      return tf::exact::projection_axes(get_pt(face[0]), get_pt(face[1]),
                                        get_pt(face[2]));
    };
    auto axes = tag == 0 ? make_axes(polygons0.faces()[object])
                         : make_axes(polygons1.faces()[object]);
    return [axes, &converter, ipts, tag, &polygons0,
            &polygons1](const auto &v) -> tf::point<Int, 2> {
      tf::point<Int, 3> pt;
      if (v.source == tf::intersect::graph::vertex_source::original) {
        if (tag == 0)
          pt = converter.convert(tf::transformed(polygons0.points()[v.id],
                                                 tf::frame_of(polygons0)));
        else
          pt = converter.convert(tf::transformed(polygons1.points()[v.id],
                                                 tf::frame_of(polygons1)));
      } else {
        pt = ipts[v.id];
      }
      return {pt[axes.first], pt[axes.second]};
    };
  };

  auto map_vertex = [&](auto tag, const auto &v) {
    return map_data.map_vertex(tag, v);
  };

  tf::buffer<Index> tri_data0;
  tf::buffer<Index> tri_data1;
  tf::buffer<Index> tri_origins0;
  tf::buffer<Index> tri_origins1;

  {
    auto selected0 =
        tf::make_indirect_range(pids[0].cut_faces[include_labels[0]],
                                tf::zip(descs_per_tag[0], loops_per_tag[0]));
    auto selected1 =
        tf::make_indirect_range(pids[1].cut_faces[include_labels[1]],
                                tf::zip(descs_per_tag[1], loops_per_tag[1]));

    tbb::parallel_invoke(
        [&] {
          tf::cut::triangulate_partition_cuts<Int>(selected0, make_projector,
                                                   map_vertex, tri_data0,
                                                   tri_origins0);
        },
        [&] {
          tf::cut::triangulate_partition_cuts<Int>(selected1, make_projector,
                                                   map_vertex, tri_data1,
                                                   tri_origins1);
        });
  }

  auto triangles0 = tf::make_blocked_range<3>(tf::make_range(tri_data0));
  auto triangles1 = tf::make_blocked_range<3>(tf::make_range(tri_data1));

  // 5. Build uncut face ranges (remapped vertex IDs)
  auto original_maps = tf::make_offset_block_range(map_data.point_offsets,
                                                   map_data.original_map);

  auto polys0 = pids[0].polygons[include_labels[0]];
  auto polys1 = pids[1].polygons[include_labels[1]];

  auto mapped_faces0 = tf::make_indirect_range(
      polys0, tf::make_block_indirect_range(
                  polygons0.faces(),
                  tf::make_mapped_range(original_maps[0],
                                        [off = map_data.original_offsets[0]](
                                            Index x) { return x + off; })));
  auto mapped_faces1 = tf::make_indirect_range(
      polys1, tf::make_block_indirect_range(
                  polygons1.faces(),
                  tf::make_mapped_range(original_maps[1],
                                        [off = map_data.original_offsets[1]](
                                            Index x) { return x + off; })));

  // 6. Concatenate faces with direction
  auto [direction0, direction1] = tf::make_directions(classes[0], classes[1]);
  auto faces = tf::core::concatenated_blocked_ranges_directed<Index>(
      std::make_pair(tf::make_range(mapped_faces0), direction0),
      std::make_pair(tf::make_range(mapped_faces1), direction1),
      std::make_pair(tf::make_range(triangles0), direction0),
      std::make_pair(tf::make_range(triangles1), direction1));

  // 7. Build points: selected originals per mesh + selected created
  auto total_pts =
      map_data.total_original_points + map_data.total_created_points;
  tf::points_buffer<RealType, 3> pts_buf;
  pts_buf.allocate(total_pts);

  {
    auto pts_range =
        tf::make_offset_block_range(map_data.original_offsets, pts_buf);
    tbb::task_group tg;
    tg.run([&] {
      auto frame = tf::frame_of(polygons0);
      tf::parallel_copy(
          tf::make_points(tf::make_indirect_range(
              map_data.original_ids[0],
              tf::make_mapped_range(
                  polygons0.points(),
                  [frame](auto pt) { return tf::transformed(pt, frame); }))),
          pts_range[0]);
    });
    tg.run([&] {
      auto frame = tf::frame_of(polygons1);
      tf::parallel_copy(
          tf::make_points(tf::make_indirect_range(
              map_data.original_ids[1],
              tf::make_mapped_range(
                  polygons1.points(),
                  [frame](auto pt) { return tf::transformed(pt, frame); }))),
          pts_range[1]);
    });
    tg.run([&] {
      tf::parallel_copy(tf::make_points(tf::make_mapped_range(
                            map_data.created_ids,
                            [&converter, &ipts](Index id) {
                              return converter.deconvert(ipts[id]);
                            })),
                        tf::drop(pts_buf, map_data.total_original_points));
    });
    tg.wait();
  }

  // 8. Build labels (which mesh each face came from)
  tf::buffer<std::int8_t> labels;
  labels.allocate(faces.size());
  auto off = std::size_t(0);
  tf::parallel_fill(tf::take(tf::drop(labels, off), mapped_faces0.size()),
                    std::int8_t(0));
  off += mapped_faces0.size();
  tf::parallel_fill(tf::take(tf::drop(labels, off), mapped_faces1.size()),
                    std::int8_t(1));
  off += mapped_faces1.size();
  tf::parallel_fill(tf::take(tf::drop(labels, off), triangles0.size()),
                    std::int8_t(0));
  off += triangles0.size();
  tf::parallel_fill(tf::take(tf::drop(labels, off), triangles1.size()),
                    std::int8_t(1));

  // Build face_labels: which original face each output face came from
  tf::buffer<Index> face_labels;
  face_labels.allocate(faces.size());
  {
    auto fl_off = std::size_t(0);
    tf::parallel_copy(polys0, tf::take(tf::drop(face_labels, fl_off),
                                        mapped_faces0.size()));
    fl_off += mapped_faces0.size();
    tf::parallel_copy(polys1, tf::take(tf::drop(face_labels, fl_off),
                                        mapped_faces1.size()));
    fl_off += mapped_faces1.size();
    tf::parallel_copy(tri_origins0, tf::take(tf::drop(face_labels, fl_off),
                                              triangles0.size()));
    fl_off += triangles0.size();
    tf::parallel_copy(tri_origins1, tf::take(tf::drop(face_labels, fl_off),
                                              triangles1.size()));
  }

  if constexpr (MakeMaps) {
    // Build index maps for tracing output back to input
    auto make_point_map = [](auto &ids_buf, auto total) {
      tf::index_map_buffer<Index> im;
      im.f().allocate(total);
      tf::parallel_fill(im.f(), Index(total));
      for (auto [i, id] : tf::enumerate(ids_buf))
        im.f()[id] = i;
      im.kept_ids() = std::move(ids_buf);
      return im;
    };

    auto original_im0 =
        make_point_map(map_data.original_ids[0],
                       map_data.point_offsets[1] - map_data.point_offsets[0]);
    auto original_im1 =
        make_point_map(map_data.original_ids[1],
                       map_data.point_offsets[2] - map_data.point_offsets[1]);

    tf::index_map_buffer<Index> created_im;
    created_im.f() = std::move(map_data.created_map);
    created_im.kept_ids() = std::move(map_data.created_ids);

    tf::index_map_buffer<Index> polygons_im0;
    tf::index_map_buffer<Index> polygons_im1;
    tf::ids_to_index_map(polys0, polygons_im0, Index(polygons0.size()),
                         Index(0), Index(polygons0.size()));
    tf::ids_to_index_map(polys1, polygons_im1, Index(polygons1.size()),
                         Index(0), Index(polygons1.size()));

    return std::make_tuple(
        tf::make_polygons_buffer(std::move(faces), std::move(pts_buf)),
        std::move(labels), std::move(face_labels),
        tf::stitch_index_maps<Index>{
            std::move(original_im0), Index(0), std::move(original_im1),
            Index(map_data.original_offsets[1]), std::move(created_im),
            Index(map_data.total_original_points), std::move(polygons_im0),
            Index(0), std::move(polygons_im1), Index(mapped_faces0.size()),
            direction0, direction1});
  } else {
    return std::make_tuple(
        tf::make_polygons_buffer(std::move(faces), std::move(pts_buf)),
        std::move(labels), std::move(face_labels));
  }
}

/// Convenience overload: make_boolean without index maps.
template <typename LabelType, typename Index, typename Policy0,
          typename Policy1, typename RealType, typename Int>
auto make_boolean(
    const tf::polygons<Policy0> &polygons0,
    const tf::polygons<Policy1> &polygons1,
    const tf::intersection_graph<Index, Int> &ig,
    const tf::face_cuts<Index, Int> &fc, const tf::cut_graph<Index> &cg,
    const tf::exact::vertex_converter<Int, RealType, 3> &converter,
    std::array<tf::arrangement_class, 2> classes,
    const tf::boolean_config &config, tf::return_index_map_t) {
  return make_boolean<LabelType>(polygons0, polygons1, ig, fc, cg,
                                        converter, classes, config,
                                        std::true_type{});
}

} // namespace tf::cut

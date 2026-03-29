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
#include "../../core/views/block_indirect_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../exact/projection_axes.hpp"
#include "../../exact/vertex_converter.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
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

template <typename LabelType, typename Index, typename Policy0,
          typename Policy1, typename RealType>
auto make_boolean_pair(const tf::polygons<Policy0> &polygons0,
                       const tf::polygons<Policy1> &polygons1,
                       const tf::intersection_graph<Index> &ig,
                       const tf::face_cuts<Index> &fc,
                       const tf::cut_graph<Index> &cg,
                       const tf::exact::vertex_converter<RealType, 3> &converter,
                       std::array<tf::arrangement_class, 2> classes,
                       const tf::boolean_config &config) {
  auto ipts = ig.points();

  auto cls_labels = tf::cut::make_classification_labels<LabelType, Index>(
      polygons0, polygons1, fc, cg, converter, tf::make_points(ipts), classes,
      config);

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

  auto map_data = tf::cut::make_splice_map_data<Index>(
      fc, static_cast<Index>(ipts.size()), pids, include_labels,
      apply_to_polygons);

  auto descs_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.descriptors());
  auto loops_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.loops());

  auto make_projector = [&](const auto &desc) {
    auto tag = desc.tag;
    auto object = desc.object;
    auto get_pt = [&, tag](Index vid) -> tf::point<int32_t, 3> {
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
            &polygons1](const auto &v) -> tf::point<int32_t, 2> {
      tf::point<int32_t, 3> pt;
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

  tf::buffer<Index> tri_data0, tri_data1;

  {
    auto selected0 =
        tf::make_indirect_range(pids[0].cut_faces[include_labels[0]],
                                tf::zip(descs_per_tag[0], loops_per_tag[0]));
    auto selected1 =
        tf::make_indirect_range(pids[1].cut_faces[include_labels[1]],
                                tf::zip(descs_per_tag[1], loops_per_tag[1]));
    tbb::parallel_invoke(
        [&] {
          tf::cut::triangulate_partition_cuts<Index>(
              selected0, make_projector, map_vertex, tri_data0);
        },
        [&] {
          tf::cut::triangulate_partition_cuts<Index>(
              selected1, make_projector, map_vertex, tri_data1);
        });
  }

  auto triangles0 = tf::make_blocked_range<3>(tf::make_range(tri_data0));
  auto triangles1 = tf::make_blocked_range<3>(tf::make_range(tri_data1));

  auto original_maps = tf::make_offset_block_range(map_data.point_offsets,
                                                   map_data.original_map);

  auto [direction0, direction1] = tf::make_directions(classes[0], classes[1]);

  auto build_half = [&](auto tag, const auto &polygons, const auto &polys,
                         const auto &triangles, const auto &original_map_range,
                         tf::direction direction) {
    auto mapped_faces = tf::make_indirect_range(
        polys, tf::make_block_indirect_range(polygons.faces(),
                                             original_map_range));

    auto faces = tf::core::concatenated_blocked_ranges_directed<Index>(
        std::make_pair(tf::make_range(mapped_faces), direction),
        std::make_pair(tf::make_range(triangles), direction));

    auto n_orig = static_cast<Index>(map_data.original_ids[tag].size());
    auto n_created = map_data.created_counts[tag];
    tf::points_buffer<RealType, 3> pts_buf;
    pts_buf.allocate(n_orig + n_created);

    auto frame = tf::frame_of(polygons);
    tf::parallel_copy(
        tf::make_points(tf::make_indirect_range(
            map_data.original_ids[tag],
            tf::make_mapped_range(
                polygons.points(),
                [frame](auto pt) { return tf::transformed(pt, frame); }))),
        tf::take(pts_buf, n_orig));

    tf::parallel_copy(
        tf::make_points(tf::make_mapped_range(
            map_data.created_ids[tag],
            [&converter, &ipts](Index id) {
              return converter.deconvert(ipts[id]);
            })),
        tf::drop(pts_buf, n_orig));

    return tf::make_polygons_buffer(std::move(faces), std::move(pts_buf));
  };

  auto polys0 = pids[0].polygons[include_labels[0]];
  auto polys1 = pids[1].polygons[include_labels[1]];

  using result_t =
      decltype(build_half(0, polygons0, polys0, triangles0, original_maps[0],
                          direction0));
  result_t left, right;

  tbb::parallel_invoke(
      [&, &direction0 = direction0] {
        left = build_half(0, polygons0, polys0, triangles0, original_maps[0],
                          direction0);
      },
      [&, &direction1 = direction1] {
        right = build_half(1, polygons1, polys1, triangles1, original_maps[1],
                           direction1);
      });

  return std::make_pair(std::move(left), std::move(right));
}

} // namespace tf::cut

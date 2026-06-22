/*
 * Copyright (c) 2026 XLAB
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
#include "../../core/algorithm/parallel_copy_blocked.hpp"
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/frame_of.hpp"
#include "../../core/memory.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/slice.hpp"
#include "../../core/views/zip.hpp"
#include "../../cut/construct/triangulate_arrangement_cuts.hpp"
#include "../../cut/face_cuts.hpp"
#include "../../exact/projection_axes.hpp"
#include "../../exact/vertex_converter.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
#include "./make_csg_map_data.hpp"
#include "./make_csg_partition.hpp"
#include "tbb/task_group.h"
#include <array>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace tf::csg::graph {

/// @ingroup csg
/// @brief Build a CSG result mesh from the implicit N-form
///        arrangement and a precomputed `chosen_sides` selection.
///
/// Assumes all input forms are triangulated (faces have 3 vertices).
/// Output: triangulated `polygons_buffer<Index, RealOut, 3, 3>`
/// whose boundary is the solid of the boolean expression that
/// produced `chosen_sides`.
template <typename OutputCoordinateType = tf::none_t, typename FormsRange,
          typename Index, typename Int, typename RealType>
auto make_csg_mesh(const tf::arrangement_graph<Index> &ag,
                   const tf::face_cuts<Index, Int> &fc,
                   const tf::intersection_graph<Index, Int> &ig,
                   const FormsRange &forms,
                   const tf::buffer<std::int8_t> &chosen_sides,
                   const tf::exact::vertex_converter<Int, RealType, 3> &conv) {
  using InputReal = tf::coordinate_type<decltype(forms[0])>;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;

  const Index n_tags = static_cast<Index>(forms.size());
  auto apply_to_polygons = [&](Index t, const auto &f) { f(forms[t]); };
  auto ipts = ig.points();

  // ---- Stages 1 + 2: partition + vertex remap. ----------------------
  auto pids = tf::csg::graph::make_csg_partition(ag, fc, forms, chosen_sides);
  auto map_data = tf::csg::graph::make_csg_map_data(
      fc, static_cast<Index>(ipts.size()), pids, apply_to_polygons);

  // ---- Stage 3: triangulate selected cut loops per (form, label). ---
  auto descs_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.descriptors());
  auto loops_per_tag =
      tf::make_offset_block_range(fc.tag_offsets(), fc.loops());

  auto make_projector = [&](const auto &desc) {
    auto tag = desc.tag;
    auto object = desc.object;
    auto face = forms[tag].faces()[object];
    auto get_pt = [&, tag](Index vid) -> tf::point<Int, 3> {
      return conv.convert(
          tf::transformed(forms[tag].points()[vid], tf::frame_of(forms[tag])));
    };
    auto axes = tf::exact::projection_axes(get_pt(face[0]), get_pt(face[1]),
                                            get_pt(face[2]));
    return [axes, ipts, tag, &forms, &conv](const auto &v) -> tf::point<Int, 2> {
      tf::point<Int, 3> pt;
      if (v.source == tf::intersect::graph::vertex_source::original)
        pt = conv.convert(
            tf::transformed(forms[tag].points()[v.id], tf::frame_of(forms[tag])));
      else
        pt = ipts[v.id];
      return {pt[axes.first], pt[axes.second]};
    };
  };

  auto map_vertex = [&](auto tag, const auto &v) {
    return map_data.map_vertex(tag, v);
  };

  // tri_data[t][L], tri_origins[t][L]: triangulation output per
  // (form, label). L = 0 reverse, L = 1 forward.
  tf::core::std_vector<std::array<tf::buffer<Index>, 2>> tri_data(n_tags);
  tf::core::std_vector<std::array<tf::buffer<Index>, 2>> tri_origins(n_tags);

  {
    tbb::task_group tg;
    for (Index t = 0; t < n_tags; ++t) {
      for (Index L = 0; L < 2; ++L) {
        tg.run([&, t, L] {
          auto selected = tf::make_indirect_range(
              pids[t].cut_faces[L],
              tf::zip(descs_per_tag[t], loops_per_tag[t]));
          tf::cut::triangulate_partition_cuts<Int>(
              selected, make_projector, map_vertex, tri_data[t][L],
              tri_origins[t][L]);
        });
      }
    }
    tg.wait();
  }

  // ---- Stage 4: per-form remapped face range view. ------------------
  auto original_maps = tf::make_offset_block_range(map_data.point_offsets,
                                                    map_data.original_map);
  auto mapped_uncut_faces = [&](Index t, Index L) {
    return tf::make_indirect_range(
        pids[t].polygons[L],
        tf::make_block_indirect_range(
            forms[t].faces(),
            tf::make_mapped_range(
                original_maps[t],
                [off = map_data.original_offsets[t]](Index x) {
                  return x + off;
                })));
  };

  // ---- Stage 5: concatenate face streams with directions. -----------
  //
  // Per form, four streams: uncut-fwd, uncut-rev, tri-fwd, tri-rev.
  // Total triangles in output = sum over (t, kind) of stream sizes.
  Index total_tris = 0;
  for (Index t = 0; t < n_tags; ++t) {
    total_tris += static_cast<Index>(pids[t].polygons[1].size());
    total_tris += static_cast<Index>(pids[t].polygons[0].size());
    total_tris += static_cast<Index>(tri_data[t][1].size() / 3);
    total_tris += static_cast<Index>(tri_data[t][0].size() / 3);
  }

  tf::blocked_buffer<Index, 3> face_buf;
  face_buf.data_buffer().allocate(3 * static_cast<std::size_t>(total_tris));

  const Index total_pts =
      map_data.total_original_points + map_data.total_created_points;
  tf::points_buffer<RealOut, 3> pts_buf;
  pts_buf.allocate(total_pts);

  {
    tbb::task_group tg;

    // ---- Stage 5: face streams. ------------------------------------
    Index offset = 0;
    auto copy_stream = [&](auto &&src, Index size, bool reverse) {
      auto dst = tf::slice(face_buf, offset, offset + size);
      if (reverse) {
        tg.run([src, dst] {
          tf::parallel_copy_blocked_reverse(src, dst);
        });
      } else {
        tg.run([src, dst] { tf::parallel_copy(src, dst); });
      }
      offset += size;
    };
    for (Index t = 0; t < n_tags; ++t) {
      copy_stream(mapped_uncut_faces(t, 1),
                  static_cast<Index>(pids[t].polygons[1].size()), false);
      copy_stream(mapped_uncut_faces(t, 0),
                  static_cast<Index>(pids[t].polygons[0].size()), true);
      copy_stream(tf::make_blocked_range<3>(tf::make_range(tri_data[t][1])),
                  static_cast<Index>(tri_data[t][1].size() / 3), false);
      copy_stream(tf::make_blocked_range<3>(tf::make_range(tri_data[t][0])),
                  static_cast<Index>(tri_data[t][0].size() / 3), true);
    }

    // ---- Stage 6: points buffer. -----------------------------------
    auto pts_range =
        tf::make_offset_block_range(map_data.original_offsets, pts_buf);
    for (Index t = 0; t < n_tags; ++t) {
      tg.run([&, t] {
        auto frame = tf::frame_of(forms[t]);
        if constexpr (std::is_integral_v<RealOut>) {
          tf::parallel_copy(
              tf::make_points(tf::make_indirect_range(
                  map_data.original_ids[t],
                  tf::make_mapped_range(
                      forms[t].points(),
                      [frame, &conv](auto pt) {
                        return conv.convert(tf::transformed(pt, frame));
                      }))),
              pts_range[t]);
        } else {
          tf::parallel_copy(
              tf::make_points(tf::make_indirect_range(
                  map_data.original_ids[t],
                  tf::make_mapped_range(
                      forms[t].points(),
                      [frame](auto pt) { return tf::transformed(pt, frame); }))),
              pts_range[t]);
        }
      });
    }
    tg.run([&] {
      if constexpr (std::is_integral_v<RealOut>) {
        tf::parallel_copy(
            tf::make_points(tf::make_mapped_range(
                map_data.created_ids,
                [&ipts](Index id) { return ipts[id]; })),
            tf::drop(pts_buf, map_data.total_original_points));
      } else {
        tf::parallel_copy(
            tf::make_points(tf::make_mapped_range(
                map_data.created_ids,
                [&conv, &ipts](Index id) {
                  return conv.deconvert(ipts[id]);
                })),
            tf::drop(pts_buf, map_data.total_original_points));
      }
    });

    tg.wait();
  }

  return tf::make_polygons_buffer(std::move(face_buf), std::move(pts_buf));
}

} // namespace tf::csg::graph

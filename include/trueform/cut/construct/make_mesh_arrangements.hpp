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
#include "../../core/polygons_buffer.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../exact/projection_axes.hpp"
#include "../../exact/vertex_converter.hpp"
#include "../../intersect/graph/intersection_graph.hpp"
#include "../face_cuts.hpp"
#include "./make_arrangement_map_data.hpp"
#include "../make_coplanar_loop_pairs.hpp"
#include "./triangulate_arrangement_cuts.hpp"

namespace tf::cut {

/// Build a mesh arrangement from a range of tagged polygon forms.
///
/// Takes the intersection graph, face cuts, and a range of polygon forms.
/// Returns (mesh, tag_labels, face_labels, map_data).
template <typename OutputCoordinateType = tf::none_t, typename Index,
          typename FormsRange, typename RealType, typename Int>
auto make_mesh_arrangements(
    const tf::intersection_graph<Index, Int> &ig,
    const tf::face_cuts<Index, Int> &fc, const FormsRange &forms,
    const tf::exact::vertex_converter<Int, RealType, 3> &converter) {
  using InputReal = tf::coordinate_type<decltype(forms[0])>;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;
  static_assert(std::is_floating_point_v<RealOut> ||
                    std::is_integral_v<RealOut>,
                "Output coordinate type must be floating-point or integral");
  static_assert(!std::is_integral_v<InputReal> ||
                    !std::is_floating_point_v<RealOut>,
                "Integer input cannot produce floating-point output");
  auto n_meshes = static_cast<Index>(forms.size());
  auto apply_to_polygons = [&](Index tag, const auto &f) { f(forms[tag]); };

  // 1. Build maps
  auto map_data =
      tf::cut::make_arrangement_map_data(fc, apply_to_polygons, n_meshes);

  // 2. Triangulate cut faces
  tf::buffer<Index> tri_data;
  tf::buffer<Index> tri_tags;
  tf::buffer<Index> tri_origins;

  auto make_projector = [&](const auto &desc) {
    auto tag = desc.tag;
    auto object = desc.object;
    auto face = forms[tag].faces()[object];
    auto ipts = ig.points();
    auto get_pt = [&, tag](Index vid) -> tf::point<Int, 3> {
      return converter.convert(
          tf::transformed(forms[tag].points()[vid], tf::frame_of(forms[tag])));
    };
    auto axes = tf::exact::projection_axes(get_pt(face[0]), get_pt(face[1]),
                                           get_pt(face[2]));
    return [axes, &converter, ipts, tag,
            &forms](const auto &v) -> tf::point<Int, 2> {
      tf::point<Int, 3> pt;
      if (v.source == tf::intersect::graph::vertex_source::original)
        pt = converter.convert(tf::transformed(forms[tag].points()[v.id],
                                               tf::frame_of(forms[tag])));
      else
        pt = ipts[v.id];
      return {pt[axes.first], pt[axes.second]};
    };
  };

  // Coincident stacks triangulate once: dead loops re-emit the
  // survivor's triangulation, winding-flipped when opposing.
  auto fold_of = tf::cut::make_loop_fold_map(
      tf::cut::make_coplanar_loop_pairs_all(fc), fc.loops().size());
  tf::cut::triangulate_arrangement_cuts<Int>(
      fc.descriptors(), fc.loops(), make_projector,
      [&](auto tag, const auto &v) { return map_data.map_vertex(tag, v); },
      fold_of, tri_data, tri_tags, tri_origins);

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
  auto faces = tf::concatenated_blocked_range_collections<Index>(
      uncut_faces, tf::make_range(&triangles, 1));

  // 5. Build points (parallel per mesh + intersection points)
  auto total_pts =
      map_data.total_original_points + static_cast<Index>(ig.points().size());
  tf::points_buffer<RealOut, 3> pts_buf;
  pts_buf.allocate(total_pts);

  {
    auto pts_range =
        tf::make_offset_block_range(map_data.original_offsets, pts_buf);
    tbb::task_group tg;
    for (Index t = 0; t < n_meshes; ++t) {
      tg.run([&, t] {
        auto frame = tf::frame_of(forms[t]);
        if constexpr (std::is_integral_v<RealOut>) {
          tf::parallel_copy(
              tf::make_points(tf::make_indirect_range(
                  map_data.original_ids[t],
                  tf::make_mapped_range(
                      forms[t].points(),
                      [frame, &converter](auto pt) {
                        return converter.convert(tf::transformed(pt, frame));
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
      auto ipts = ig.points();
      if constexpr (std::is_integral_v<RealOut>) {
        tf::parallel_copy(tf::make_points(ipts),
                          tf::drop(pts_buf, map_data.total_original_points));
      } else {
        tf::parallel_copy(
            tf::make_points(tf::make_mapped_range(
                ipts,
                [&converter](auto pt) { return converter.deconvert(pt); })),
            tf::drop(pts_buf, map_data.total_original_points));
      }
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
    auto tag_uncut =
        tf::make_offset_block_range(map_data.original_face_offsets, tag_labels);
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
      std::move(tag_labels), std::move(face_labels), std::move(map_data));
}

/// 2-mesh overload: different policy types.
template <typename OutputCoordinateType = tf::none_t, typename Index,
          typename Policy0, typename Policy1, typename RealType, typename Int>
auto make_mesh_arrangements(
    const tf::intersection_graph<Index, Int> &ig,
    const tf::face_cuts<Index, Int> &fc, const tf::polygons<Policy0> &form0,
    const tf::polygons<Policy1> &form1,
    const tf::exact::vertex_converter<Int, RealType, 3> &converter) {
  using InputReal = tf::coordinate_type<Policy0, Policy1>;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;
  static_assert(std::is_floating_point_v<RealOut> ||
                    std::is_integral_v<RealOut>,
                "Output coordinate type must be floating-point or integral");
  static_assert(!std::is_integral_v<InputReal> ||
                    !std::is_floating_point_v<RealOut>,
                "Integer input cannot produce floating-point output");
  auto apply_to_polygons = [&](Index tag, const auto &f) {
    if (tag == 0)
      f(form0);
    else
      f(form1);
  };

  auto map_data =
      tf::cut::make_arrangement_map_data(fc, apply_to_polygons, Index(2));

  tf::buffer<Index> tri_data;
  tf::buffer<Index> tri_tags;
  tf::buffer<Index> tri_origins;

  auto make_projector = [&](const auto &desc) {
    auto tag = desc.tag;
    auto object = desc.object;
    auto ipts = ig.points();
    auto get_pt = [&, tag](Index vid) -> tf::point<Int, 3> {
      if (tag == 0)
        return converter.convert(
            tf::transformed(form0.points()[vid], tf::frame_of(form0)));
      else
        return converter.convert(
            tf::transformed(form1.points()[vid], tf::frame_of(form1)));
    };
    auto make_axes = [&](auto face) {
      return tf::exact::projection_axes(get_pt(face[0]), get_pt(face[1]),
                                        get_pt(face[2]));
    };
    auto axes = tag == 0 ? make_axes(form0.faces()[object])
                         : make_axes(form1.faces()[object]);
    return [axes, &converter, ipts, tag, &form0,
            &form1](const auto &v) -> tf::point<Int, 2> {
      tf::point<Int, 3> pt;
      if (v.source == tf::intersect::graph::vertex_source::original) {
        if (tag == 0)
          pt = converter.convert(
              tf::transformed(form0.points()[v.id], tf::frame_of(form0)));
        else
          pt = converter.convert(
              tf::transformed(form1.points()[v.id], tf::frame_of(form1)));
      } else {
        pt = ipts[v.id];
      }
      return {pt[axes.first], pt[axes.second]};
    };
  };

  // Coincident stacks triangulate once: dead loops re-emit the
  // survivor's triangulation, winding-flipped when opposing.
  auto fold_of = tf::cut::make_loop_fold_map(
      tf::cut::make_coplanar_loop_pairs_all(fc), fc.loops().size());
  tf::cut::triangulate_arrangement_cuts<Int>(
      fc.descriptors(), fc.loops(), make_projector,
      [&](auto tag, const auto &v) { return map_data.map_vertex(tag, v); },
      fold_of, tri_data, tri_tags, tri_origins);

  auto triangles = tf::make_blocked_range<3>(tf::make_range(tri_data));

  auto original_maps = tf::make_offset_block_range(map_data.point_offsets,
                                                   map_data.original_map);

  auto uncut_faces0 = tf::make_indirect_range(
      map_data.original_face_ids[0],
      tf::make_block_indirect_range(
          form0.faces(),
          tf::make_mapped_range(original_maps[0],
                                [off = map_data.original_offsets[0]](Index x) {
                                  return x + off;
                                })));
  auto uncut_faces1 = tf::make_indirect_range(
      map_data.original_face_ids[1],
      tf::make_block_indirect_range(
          form1.faces(),
          tf::make_mapped_range(original_maps[1],
                                [off = map_data.original_offsets[1]](Index x) {
                                  return x + off;
                                })));

  auto faces = tf::concatenated_blocked_range_collections<Index>(
      tf::make_range(&uncut_faces0, 1), tf::make_range(&uncut_faces1, 1),
      tf::make_range(&triangles, 1));

  auto total_pts =
      map_data.total_original_points + static_cast<Index>(ig.points().size());
  tf::points_buffer<RealOut, 3> pts_buf;
  pts_buf.allocate(total_pts);

  {
    auto pts_range =
        tf::make_offset_block_range(map_data.original_offsets, pts_buf);
    tbb::task_group tg;
    tg.run([&] {
      auto frame = tf::frame_of(form0);
      if constexpr (std::is_integral_v<RealOut>) {
        tf::parallel_copy(
            tf::make_points(tf::make_indirect_range(
                map_data.original_ids[0],
                tf::make_mapped_range(
                    form0.points(),
                    [frame, &converter](auto pt) {
                      return converter.convert(tf::transformed(pt, frame));
                    }))),
            pts_range[0]);
      } else {
        tf::parallel_copy(
            tf::make_points(tf::make_indirect_range(
                map_data.original_ids[0],
                tf::make_mapped_range(
                    form0.points(),
                    [frame](auto pt) { return tf::transformed(pt, frame); }))),
            pts_range[0]);
      }
    });
    tg.run([&] {
      auto frame = tf::frame_of(form1);
      if constexpr (std::is_integral_v<RealOut>) {
        tf::parallel_copy(
            tf::make_points(tf::make_indirect_range(
                map_data.original_ids[1],
                tf::make_mapped_range(
                    form1.points(),
                    [frame, &converter](auto pt) {
                      return converter.convert(tf::transformed(pt, frame));
                    }))),
            pts_range[1]);
      } else {
        tf::parallel_copy(
            tf::make_points(tf::make_indirect_range(
                map_data.original_ids[1],
                tf::make_mapped_range(
                    form1.points(),
                    [frame](auto pt) { return tf::transformed(pt, frame); }))),
            pts_range[1]);
      }
    });
    tg.run([&] {
      auto ipts = ig.points();
      if constexpr (std::is_integral_v<RealOut>) {
        tf::parallel_copy(tf::make_points(ipts),
                          tf::drop(pts_buf, map_data.total_original_points));
      } else {
        tf::parallel_copy(
            tf::make_points(tf::make_mapped_range(
                ipts,
                [&converter](auto pt) { return converter.deconvert(pt); })),
            tf::drop(pts_buf, map_data.total_original_points));
      }
    });
    tg.wait();
  }

  auto total_faces = static_cast<Index>(faces.size());
  tf::buffer<Index> tag_labels;
  tf::buffer<Index> face_labels;
  tag_labels.allocate(total_faces);
  face_labels.allocate(total_faces);

  {
    auto tag_uncut =
        tf::make_offset_block_range(map_data.original_face_offsets, tag_labels);
    auto face_uncut = tf::make_offset_block_range(
        map_data.original_face_offsets, face_labels);
    tbb::parallel_invoke(
        [&] {
          tf::parallel_fill(tag_uncut[0], Index(0));
          tf::parallel_copy(map_data.original_face_ids[0], face_uncut[0]);
        },
        [&] {
          tf::parallel_fill(tag_uncut[1], Index(1));
          tf::parallel_copy(map_data.original_face_ids[1], face_uncut[1]);
        },
        [&] {
          auto tri_off = map_data.total_original_faces;
          tf::parallel_copy(tri_tags, tf::drop(tag_labels, tri_off));
          tf::parallel_copy(tri_origins, tf::drop(face_labels, tri_off));
        });
  }

  return std::make_tuple(
      tf::make_polygons_buffer(std::move(faces), std::move(pts_buf)),
      std::move(tag_labels), std::move(face_labels), std::move(map_data));
}

} // namespace tf::cut

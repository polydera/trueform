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
#include "../../exact/vertex_converter.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../region_triangulator.hpp"
#include "./make_arrangement_map_data.hpp"

namespace tf::cut {

template <typename OutputCoordinateType = tf::none_t, typename Index,
          typename Policy, typename RealType, typename Int>
auto make_polygon_arrangements(
    const tf::polygons<Policy> &polygons,
    const tf::cut::region_triangulator<Index, Int> &rt,
    const tf::exact::vertex_converter<Int, RealType, 3> &converter,
    const tf::buffer<tf::point<Int, 3>> &created_pts) {
  using InputReal = tf::coordinate_type<Policy>;
  using RealOut =
      std::conditional_t<std::is_same_v<OutputCoordinateType, tf::none_t>,
                         InputReal, OutputCoordinateType>;
  static_assert(std::is_floating_point_v<RealOut> ||
                    std::is_integral_v<RealOut>,
                "Output coordinate type must be floating-point or integral");
  static_assert(!std::is_integral_v<InputReal> ||
                    !std::is_floating_point_v<RealOut>,
                "Integer input cannot produce floating-point output");
  auto map_data = tf::cut::make_embed_map_data(
      rt, polygons, Index(0), static_cast<Index>(created_pts.size()));

  // Cut loops: one exposed triangle each, in loop order — promoted
  // (conforming) faces carry descriptors, so they are already out of
  // the uncut list and ride this stream.
  tf::buffer<Index> tri_data;
  tf::buffer<Index> tri_origins;
  {
    auto loops = rt.loops();
    auto descs = rt.descriptors();
    const std::size_t n_tris = std::size_t(loops.size());
    tri_data.allocate(3 * n_tris);
    tri_origins.allocate(n_tris);
    tf::parallel_for_each(
        tf::make_sequence_range(n_tris), [&](std::size_t l) {
          const auto &tr = loops[Index(l)];
          for (int c = 0; c < 3; ++c)
            tri_data[3 * l + std::size_t(c)] =
                map_data.map_vertex(tr[std::size_t(c)]);
          tri_origins[l] = descs[Index(l)].object;
        });
  }

  auto triangles = tf::make_blocked_range<3>(tf::make_range(tri_data));

  auto mapped_faces = tf::make_indirect_range(
      tf::make_range(map_data.uncut_face_ids),
      tf::make_block_indirect_range(polygons.faces(), map_data.original_map));

  auto faces = tf::core::concatenated_blocked_ranges_directed<Index>(
      std::make_pair(tf::make_range(mapped_faces), tf::direction::forward),
      std::make_pair(tf::make_range(triangles), tf::direction::forward));

  tf::points_buffer<RealOut, 3> pts_buf;
  pts_buf.allocate(map_data.n_original_points + map_data.total_created_used);
  auto gathered_created = tf::make_indirect_range(
      tf::make_range(map_data.created_ids), tf::make_range(created_pts));

  auto frame = tf::frame_of(polygons);
  if constexpr (std::is_integral_v<RealOut>) {
    tf::parallel_copy(
        tf::make_points(tf::make_indirect_range(
            map_data.original_ids,
            tf::make_mapped_range(polygons.points(),
                                  [frame, &converter](auto pt) {
                                    return converter.convert(
                                        tf::transformed(pt, frame));
                                  }))),
        tf::take(pts_buf, map_data.n_original_points));
    tf::parallel_copy(tf::make_points(gathered_created),
                      tf::drop(pts_buf, map_data.n_original_points));
  } else {
    tf::parallel_copy(tf::make_points(tf::make_indirect_range(
                          map_data.original_ids,
                          tf::make_mapped_range(polygons.points(),
                                                [frame](auto pt) {
                                                  return tf::transformed(pt,
                                                                         frame);
                                                }))),
                      tf::take(pts_buf, map_data.n_original_points));
    tf::parallel_copy(
        tf::make_points(tf::make_mapped_range(
            gathered_created,
            [&converter](auto pt) { return converter.deconvert(pt); })),
        tf::drop(pts_buf, map_data.n_original_points));
  }

  // Face labels: uncut → original face ID, triangulated → desc.object
  auto n_uncut = static_cast<Index>(map_data.uncut_face_ids.size());
  tf::buffer<Index> face_labels;
  face_labels.allocate(faces.size());
  tf::parallel_copy(tf::make_range(map_data.uncut_face_ids),
                    tf::take(face_labels, n_uncut));
  tf::parallel_copy(tri_origins, tf::drop(face_labels, n_uncut));

  return std::make_tuple(
      tf::make_polygons_buffer(std::move(faces), std::move(pts_buf)),
      std::move(face_labels), std::move(map_data));
}

} // namespace tf::cut

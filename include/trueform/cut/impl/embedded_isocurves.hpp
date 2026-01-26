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
#include "../../core/concatenated_blocked_ranges.hpp"
#include "../../core/polygons.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/take.hpp"
#include "../../intersect/types/simple_intersections.hpp"
#include "../scalar_cut_faces.hpp"
#include "./scalars.hpp"
#include "./triangulate_cut_faces.hpp"

namespace tf::cut {
template <typename LabelType, typename Policy, typename Index, typename RealT,
          std::size_t Dims, typename Range, typename Iterator, std::size_t N>
auto embedded_isocurves(
    const tf::polygons<Policy> &polygons,
    const tf::intersect::simple_intersections<Index, RealT, Dims> &sfi,
    const tf::scalar_cut_faces<Index> &scf, const Range &scalars,
    tf::range<Iterator, N> cut_values) {
  auto all_ids = tf::cut::make_polygon_arrangement_ids<Index>(
      polygons, sfi, scf, scalars, cut_values);

  const Index n_orig = static_cast<Index>(polygons.points().size());
  const Index n_ip = static_cast<Index>(sfi.intersection_points().size());

  tf::points_buffer<tf::coordinate_type<Policy>, tf::coordinate_dims_v<Policy>>
      points_out;
  points_out.allocate(n_orig + n_ip);

  tf::parallel_copy(polygons.points(), tf::take(points_out, n_orig));
  tf::parallel_copy(sfi.intersection_points(), tf::drop(points_out, n_orig));

  tf::buffer<Index> cf_offsets;
  tf::blocked_buffer<Index, 3> triangles;

  auto map_vertex_f = [&](auto v) -> Index {
    return (v.source == tf::loop::vertex_source::created) ? (n_orig + v.id)
                                                          : v.id;
  };

  auto make_projector = [&](auto d) {
    auto proj = tf::make_simple_projector(tf::make_normal(polygons[d.object]));
    return [proj, &polygons, &sfi](auto v) {
      return (v.source == tf::loop::vertex_source::original)
                 ? proj(polygons.points()[v.id])
                 : proj(sfi.intersection_points()[v.id]);
    };
  };

  tf::generate_offset_blocks(
      all_ids.cut_faces, cf_offsets, triangles,
      [&](const auto &ids, auto &data) {
        tf::cut::triangulate_cut_faces(
            tf::make_indirect_range(
                ids, tf::zip(scf.descriptors(), scf.mapped_loops())),
            [&](auto d) { return make_projector(d); }, map_vertex_f,
            data.data_buffer());
      });

  auto faces = tf::concatenated_blocked_ranges<Index>(
      tf::make_indirect_range(all_ids.polygons.data_buffer(), polygons.faces()),
      triangles);

  tf::buffer<LabelType> labels;
  labels.allocate(faces.size());
  std::size_t polygon_size = all_ids.polygons.data_buffer().size();
  auto cut_labels = tf::drop(labels, polygon_size);
  auto original_labels = tf::take(labels, polygon_size);
  tf::parallel_for_each(
      tf::enumerate(tf::make_offset_block_range(cf_offsets, cut_labels)),
      [](auto pair) {
        auto [id, r] = pair;
        std::fill(r.begin(), r.end(), id);
      },
      tf::checked);
  Index start = 0;
  for (const auto &[id, r] : tf::enumerate(all_ids.polygons)) {
    Index end = start + r.size();
    tf::parallel_fill(tf::slice(original_labels, start, end), id);
    start = end;
  }
  return std::make_pair(
      tf::make_polygons_buffer(std::move(faces), std::move(points_out)),
      std::move(labels));
}

} // namespace tf::cut

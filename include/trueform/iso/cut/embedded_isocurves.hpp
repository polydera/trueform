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

#include "../../arrangement/partition/partition_ids.hpp"
#include "../../core/algorithm/parallel_copy.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/blocked_buffer.hpp"
#include "../../core/buffer.hpp"
#include "../../core/concatenated_blocked_ranges.hpp"
#include "../../core/coordinate_dims.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/enumerate.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/slice.hpp"
#include "../../core/views/take.hpp"
#include "../../intersect/records/simple_intersections.hpp"
#include "./gather_iso_band_triangles.hpp"
#include "./iso_cut_regions.hpp"

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

namespace tf::iso {

/// The whole mesh recut: every band's uncut faces keep their own arity, and
/// the pieces follow as triangles in the flat identity space the cut pass
/// already emitted.
template <typename LabelType, typename Index, typename Policy, typename RealT,
          std::size_t Dims>
auto embedded_isocurves(
    const tf::polygons<Policy> &polygons,
    const tf::intersect::simple_intersections<Index, RealT, Dims> &si,
    const iso_cut_regions<Index, RealT, Dims> &regions,
    const tf::arrangement::partition_ids<Index> &pids) {
  const Index n_original = static_cast<Index>(polygons.points().size());
  const Index n_field = static_cast<Index>(si.intersection_points().size());

  tf::points_buffer<tf::coordinate_type<Policy>, tf::coordinate_dims_v<Policy>>
      pts_buf;
  pts_buf.allocate(std::size_t(n_original) + std::size_t(n_field) +
                   regions.minted_points.size());
  tf::parallel_copy(polygons.points(), tf::take(pts_buf, n_original));
  tf::parallel_copy(si.intersection_points(),
                    tf::take(tf::drop(pts_buf, n_original), n_field));
  tf::parallel_copy(tf::make_range(regions.minted_points),
                    tf::drop(pts_buf, std::size_t(n_original) +
                                          std::size_t(n_field)));

  tf::blocked_buffer<Index, 3> triangles;
  tf::buffer<Index> tri_origins;
  tf::buffer<Index> cf_offsets;
  gather_iso_band_triangles<Index>(pids.cut_faces, regions, triangles,
                                   tri_origins, cf_offsets);

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

  tf::buffer<Index> face_labels;
  face_labels.allocate(faces.size());
  tf::parallel_copy(pids.polygons.data_buffer(),
                    tf::take(face_labels, polygon_size));
  tf::parallel_copy(tri_origins, tf::drop(face_labels, polygon_size));

  return std::make_tuple(
      tf::make_polygons_buffer(std::move(faces), std::move(pts_buf)),
      std::move(labels), std::move(face_labels));
}

} // namespace tf::iso

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
#include "../../core/concatenated_blocked_range_collections.hpp"
#include "../../core/coordinate_dims.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/views/block_indirect_range.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/slice.hpp"
#include "../../core/views/take.hpp"
#include "../../core/views/zip.hpp"
#include "../../intersect/records/simple_intersections.hpp"
#include "./gather_iso_band_triangles.hpp"
#include "./iso_cut_regions.hpp"

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <utility>

namespace tf::iso {

/// The selected bands alone. Discovery walks the kept pieces and the kept
/// uncut faces in the flat identity space, so one map compacts both; the
/// created ids it publishes are the field points, which is what the curve
/// filter names.
template <typename LabelType, typename Index, typename Policy, typename RealT,
          std::size_t Dims, typename SelectedBands>
auto make_isobands(
    const tf::polygons<Policy> &polygons,
    const tf::intersect::simple_intersections<Index, RealT, Dims> &si,
    const iso_cut_regions<Index, RealT, Dims> &regions,
    const tf::arrangement::partition_ids<Index> &pids,
    const SelectedBands &selected_bands) {
  auto polygon_ids = tf::make_indirect_range(selected_bands, pids.polygons);
  auto cut_ids = tf::make_indirect_range(selected_bands, pids.cut_faces);

  const Index n_original = static_cast<Index>(polygons.points().size());
  const Index n_field = static_cast<Index>(si.intersection_points().size());
  const Index n_minted = static_cast<Index>(regions.minted_points.size());

  tf::buffer<Index> flat_map;
  flat_map.allocate(std::size_t(n_original) + std::size_t(n_field) +
                    std::size_t(n_minted));
  const Index sentinel = static_cast<Index>(flat_map.size());
  tf::parallel_fill(flat_map, sentinel);

  tf::buffer<Index> original_ids;
  original_ids.reserve(std::size_t(n_original));
  tf::buffer<Index> created_ids;
  created_ids.reserve(std::size_t(n_field));
  tf::buffer<Index> minted_ids;

  Index n_kept_original = 0;
  Index n_kept_created = 0;
  Index n_kept_minted = 0;
  auto discover = [&](Index flat) {
    auto &slot = flat_map[std::size_t(flat)];
    if (slot != sentinel)
      return;
    if (flat < n_original) {
      slot = n_kept_original++;
      original_ids.push_back(flat);
    } else if (flat < n_original + n_field) {
      slot = n_kept_created++;
      created_ids.push_back(flat - n_original);
    } else {
      slot = n_kept_minted++;
      minted_ids.push_back(flat - n_original - n_field);
    }
  };

  tf::blocked_buffer<Index, 3> triangles;
  tf::buffer<Index> tri_origins;
  tf::buffer<Index> cf_offsets;
  gather_iso_band_triangles<Index>(cut_ids, regions, triangles, tri_origins,
                                   cf_offsets);

  for (const auto corner : triangles.data_buffer())
    discover(corner);

  for (auto band_faces :
       tf::make_block_indirect_range(polygon_ids, polygons.faces()))
    for (const auto &face : band_faces)
      for (auto v : face)
        discover(v);

  for (const auto id : created_ids)
    flat_map[std::size_t(n_original + id)] += n_kept_original;
  for (const auto id : minted_ids)
    flat_map[std::size_t(n_original + n_field + id)] +=
        n_kept_original + n_kept_created;

  auto faces = tf::concatenated_blocked_range_collections<Index>(
      tf::make_block_indirect_range(
          polygon_ids,
          tf::make_block_indirect_range(polygons.faces(), flat_map)),
      tf::make_offset_block_range(
          cf_offsets, tf::make_block_indirect_range(triangles, flat_map)));

  tf::points_buffer<tf::coordinate_type<Policy>, tf::coordinate_dims_v<Policy>>
      pts_out;
  pts_out.allocate(std::size_t(n_kept_original) + std::size_t(n_kept_created) +
                   std::size_t(n_kept_minted));
  tf::parallel_copy(tf::make_indirect_range(original_ids, polygons.points()),
                    tf::take(pts_out, n_kept_original));
  tf::parallel_copy(
      tf::make_indirect_range(created_ids, si.intersection_points()),
      tf::take(tf::drop(pts_out, n_kept_original), n_kept_created));
  tf::parallel_copy(
      tf::make_indirect_range(minted_ids, tf::make_range(regions.minted_points)),
      tf::drop(pts_out, std::size_t(n_kept_original) +
                            std::size_t(n_kept_created)));

  std::size_t polygon_size = 0;
  for (auto r : polygon_ids)
    polygon_size += r.size();

  tf::buffer<LabelType> labels;
  labels.allocate(faces.size());
  auto original_labels = tf::take(labels, polygon_size);
  auto cut_labels = tf::drop(labels, polygon_size);

  tf::parallel_for_each(
      tf::zip(selected_bands,
              tf::make_offset_block_range(cf_offsets, cut_labels)),
      [](auto pair) {
        auto [id, r] = pair;
        std::fill(r.begin(), r.end(), id);
      },
      tf::checked);

  Index start = 0;
  for (auto [id, r] : tf::zip(selected_bands, polygon_ids)) {
    Index end = start + r.size();
    tf::parallel_fill(tf::slice(original_labels, start, end), id);
    start = end;
  }

  tf::buffer<Index> face_labels;
  face_labels.allocate(faces.size());
  {
    auto fl_off = std::size_t(0);
    for (auto band_polys : polygon_ids) {
      tf::parallel_copy(band_polys, tf::take(tf::drop(face_labels, fl_off),
                                             band_polys.size()));
      fl_off += band_polys.size();
    }
    tf::parallel_copy(tri_origins, tf::drop(face_labels, fl_off));
  }

  return std::make_tuple(
      tf::make_polygons_buffer(std::move(faces), std::move(pts_out)),
      std::move(labels), std::move(face_labels), std::move(created_ids));
}

} // namespace tf::iso

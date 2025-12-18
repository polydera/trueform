/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "../core/algorithm/parallel_apply.hpp"
#include "../core/algorithm/parallel_fill.hpp"
#include "../core/buffer.hpp"
#include "../core/faces.hpp"
#include "../core/stitch_index_maps.hpp"
#include "../core/views/indirect_range.hpp"
#include "../core/views/mapped_range.hpp"
#include "../core/views/sequence_range.hpp"
#include "./face_membership.hpp"
#include "./face_membership_like.hpp"
#include <algorithm>

namespace tf {

/// Stitch face_membership from two source meshes using stitch index maps.
template <typename Index, typename FacesPolicy, typename FMPolicy0,
          typename FMPolicy1>
auto stitched_face_membership(const tf::faces<FacesPolicy> &result_faces,
                              std::size_t n_result_points,
                              const tf::face_membership_like<FMPolicy0> &fm0,
                              const tf::face_membership_like<FMPolicy1> &fm1,
                              const tf::stitch_index_maps<Index> &im)
    -> tf::face_membership<Index> {
  const Index dirty_start =
      im.polygons0.kept_ids().size() + im.polygons1.kept_ids().size();
  auto dirty_ids =
      tf::make_sequence_range(dirty_start, Index(result_faces.size()));
  auto dirty_mask =
      tf::make_mapped_range(tf::make_sequence_range(result_faces.size()),
                            [&](Index i) { return i >= dirty_start; });
  Index total_dirty_size = dirty_ids.size() * 3;

  tf::face_membership<Index> dirty_fm;
  dirty_fm.build(
      tf::make_faces(tf::make_indirect_range(dirty_ids, result_faces)),
      n_result_points, total_dirty_size);

  tf::buffer<Index> offsets;
  offsets.allocate(n_result_points + 1);
  tf::parallel_fill(offsets, 0);
  const Index num_created = n_result_points - im.created_points_offset;
  constexpr Index sentinel = Index(-1);

  // Points from mesh0: count kept clean polygons + dirty
  tf::parallel_apply(
      im.points0.kept_ids(),
      [&](Index orig_idx) {
        Index result_idx = im.points0.f()[orig_idx] + im.points0_offset;
        Index count = 0;
        for (auto poly_id : fm0[orig_idx]) {
          Index remapped = im.polygons0.f()[poly_id];
          if (remapped != sentinel &&
              !dirty_mask[remapped + im.polygons0_offset])
            ++count;
        }
        offsets[result_idx + 1] = count + dirty_fm[result_idx].size();
      },
      tf::checked);

  // Points from mesh1: count kept clean polygons + dirty
  tf::parallel_apply(
      im.points1.kept_ids(),
      [&](Index orig_idx) {
        Index result_idx = im.points1.f()[orig_idx] + im.points1_offset;
        Index count = 0;
        for (auto poly_id : fm1[orig_idx]) {
          Index remapped = im.polygons1.f()[poly_id];
          if (remapped != sentinel &&
              !dirty_mask[remapped + im.polygons1_offset])
            ++count;
        }
        offsets[result_idx + 1] = count + dirty_fm[result_idx].size();
      },
      tf::checked);

  // Created points: only dirty contribution
  tf::parallel_apply(
      tf::make_sequence_range(num_created),
      [&](Index i) {
        Index result_idx = im.created_points_offset + i;
        offsets[result_idx + 1] = dirty_fm[result_idx].size();
      },
      tf::checked);

  // Sequential prefix sum -> offsets
  for (std::size_t point_id = 0; point_id < n_result_points; ++point_id) {
    offsets[point_id + 1] += offsets[point_id];
  }

  // Allocate and copy
  tf::face_membership<Index> result;
  result.data_buffer().allocate(offsets.back());
  auto &data = result.data_buffer();
  const auto &offs = offsets;

  // Copy from mesh0: kept clean polygons (remapped) + dirty
  tf::parallel_apply(
      im.points0.kept_ids(),
      [&](Index orig_idx) {
        Index result_idx = im.points0.f()[orig_idx] + im.points0_offset;
        auto dest = data.begin() + offs[result_idx];
        for (auto poly_id : fm0[orig_idx]) {
          Index remapped = im.polygons0.f()[poly_id];
          if (remapped != sentinel &&
              !dirty_mask[remapped + im.polygons0_offset])
            *dest++ = remapped + im.polygons0_offset;
        }
        for (auto local_poly_id : dirty_fm[result_idx]) {
          *dest++ = dirty_ids[local_poly_id];
        }
        std::sort(data.begin() + offs[result_idx],
                  data.begin() + offs[result_idx + 1], std::greater<Index>());
      },
      tf::checked);

  // Copy from mesh1: kept clean polygons (remapped) + dirty
  tf::parallel_apply(
      im.points1.kept_ids(),
      [&](Index orig_idx) {
        Index result_idx = im.points1.f()[orig_idx] + im.points1_offset;
        auto dest = data.begin() + offs[result_idx];
        for (auto poly_id : fm1[orig_idx]) {
          Index remapped = im.polygons1.f()[poly_id];
          if (remapped != sentinel &&
              !dirty_mask[remapped + im.polygons1_offset])
            *dest++ = remapped + im.polygons1_offset;
        }
        for (auto local_poly_id : dirty_fm[result_idx]) {
          *dest++ = dirty_ids[local_poly_id];
        }
        std::sort(data.begin() + offs[result_idx],
                  data.begin() + offs[result_idx + 1], std::greater<Index>());
      },
      tf::checked);

  // Copy created points: only dirty polygons
  tf::parallel_apply(
      tf::make_sequence_range(num_created),
      [&](Index i) {
        Index result_idx = im.created_points_offset + i;
        auto dest = data.begin() + offs[result_idx];
        for (auto local_poly_id : dirty_fm[result_idx]) {
          *dest++ = dirty_ids[local_poly_id];
        }
        std::sort(data.begin() + offs[result_idx],
                  data.begin() + offs[result_idx + 1], std::greater<Index>());
      },
      tf::checked);

  result.offsets_buffer() = std::move(offsets);

  return result;
}

} // namespace tf

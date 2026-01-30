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
#include "../core/algorithm/parallel_for_each.hpp"
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

/// @ingroup topology_connectivity
/// @brief Stitch face membership from two source meshes using stitch index maps.
///
/// Combines face membership structures from two meshes being stitched together.
/// Uses the stitch index maps to remap vertex and face indices, and recomputes
/// connectivity for faces affected by the stitching operation.
///
/// @tparam Index The integer type for indices.
/// @tparam FacesPolicy The result faces policy type.
/// @tparam FMPolicy0 The first face membership policy type.
/// @tparam FMPolicy1 The second face membership policy type.
/// @param result_faces The faces of the stitched result mesh.
/// @param n_result_points The number of points in the result mesh.
/// @param fm0 Face membership from the first source mesh.
/// @param fm1 Face membership from the second source mesh.
/// @param im The stitch index maps for remapping.
/// @return A new face membership structure for the stitched mesh.
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
  Index total_dirty_size = [&] {
    if constexpr (tf::static_size_v<decltype(result_faces[0])> !=
                  tf::dynamic_size)
      return Index(dirty_ids.size()) *
             Index(tf::static_size_v<decltype(result_faces[0])>);
    else
      return tf::reduce(
          tf::make_mapped_range(
              tf::make_indirect_range(dirty_ids, result_faces),
              [](const auto &f) -> Index { return f.size(); }),
          std::plus<>(), Index(0), tf::checked);
  }();

  tf::face_membership<Index> dirty_fm;
  dirty_fm.build(
      tf::make_faces(tf::make_indirect_range(dirty_ids, result_faces)),
      n_result_points, total_dirty_size);

  tf::buffer<Index> offsets;
  offsets.allocate(n_result_points + 1);
  tf::parallel_fill(offsets, 0);
  const Index num_created = n_result_points - im.created_points_offset;
  const Index sentinel0 = Index(im.polygons0.f().size());
  const Index sentinel1 = Index(im.polygons1.f().size());

  // Points from mesh0: count kept clean polygons + dirty
  tf::parallel_for_each(
      im.points0.kept_ids(),
      [&, sentinel0](Index orig_idx) {
        Index result_idx = im.points0.f()[orig_idx] + im.points0_offset;
        Index count = 0;
        for (auto poly_id : fm0[orig_idx]) {
          Index remapped = im.polygons0.f()[poly_id];
          if (remapped != sentinel0 &&
              !dirty_mask[remapped + im.polygons0_offset])
            ++count;
        }
        offsets[result_idx + 1] = count + dirty_fm[result_idx].size();
      },
      tf::checked);

  // Points from mesh1: count kept clean polygons + dirty
  tf::parallel_for_each(
      im.points1.kept_ids(),
      [&, sentinel1](Index orig_idx) {
        Index result_idx = im.points1.f()[orig_idx] + im.points1_offset;
        Index count = 0;
        for (auto poly_id : fm1[orig_idx]) {
          Index remapped = im.polygons1.f()[poly_id];
          if (remapped != sentinel1 &&
              !dirty_mask[remapped + im.polygons1_offset])
            ++count;
        }
        offsets[result_idx + 1] = count + dirty_fm[result_idx].size();
      },
      tf::checked);

  // Created points: only dirty contribution
  tf::parallel_for_each(
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
  tf::parallel_for_each(
      im.points0.kept_ids(),
      [&, sentinel0](Index orig_idx) {
        Index result_idx = im.points0.f()[orig_idx] + im.points0_offset;
        auto dest = data.begin() + offs[result_idx];
        for (auto poly_id : fm0[orig_idx]) {
          Index remapped = im.polygons0.f()[poly_id];
          if (remapped != sentinel0 &&
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
  tf::parallel_for_each(
      im.points1.kept_ids(),
      [&, sentinel1](Index orig_idx) {
        Index result_idx = im.points1.f()[orig_idx] + im.points1_offset;
        auto dest = data.begin() + offs[result_idx];
        for (auto poly_id : fm1[orig_idx]) {
          Index remapped = im.polygons1.f()[poly_id];
          if (remapped != sentinel1 &&
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
  tf::parallel_for_each(
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

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
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/none.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/polygons_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/resolved_output_real.hpp"
#include "../../core/views/blocked_range.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./emit_arrangement_points.hpp"
#include "./make_arrangement_map_data.hpp"

#include "tbb/task_group.h"

#include <cstddef>
#include <tuple>
#include <utility>

namespace tf::arrangement {

/// The arrangement mesh of any operand shape. Everything here is per-tag
/// work reached through `apply_to_form`; the one step that depends on how
/// the operands are stored — assembling the output faces — is the policy's
/// `concatenated_output_faces`.
template <typename OutputCoordinateType = tf::none_t, typename Arrangement>
auto make_mesh_arrangements(const Arrangement &arrangement) {
  using Index = typename Arrangement::index_type;
  using Policy = typename Arrangement::policy_type;
  const auto &policy = arrangement.policy();
  const auto &created_pts = arrangement.created_points();
  using InputReal = typename Policy::input_real_type;
  using RealOut = tf::resolved_output_real_t<OutputCoordinateType, InputReal>;
  auto n_meshes = static_cast<Index>(policy.n_tags());
  auto apply_to_polygons = policy.make_apply_to_form();

  // The slot is the authority for what is uncut, so the map build
  // already keeps cut faces out of the uncut lists.
  auto map_data =
      tf::arrangement::make_arrangement_stream_map_data<Index>(arrangement,
                                                       apply_to_polygons);

  // Cut faces: the exposed triangle stream, in stream order — winding
  // (coplanar folds) is already applied there.
  tf::buffer<Index> tri_data;
  [[maybe_unused]] tf::buffer<Index> tri_tags;
  tf::buffer<Index> tri_origins;
  {
    auto tris = arrangement.global().exposed_tris();
    auto descs = arrangement.global().exposed_descriptors();
    auto slots = arrangement.triangle_slots();
    auto tags = arrangement.triangle_tags();
    const std::size_t n_tris = std::size_t(tris.size());
    tri_data.allocate(3 * n_tris);
    tri_tags.allocate(n_tris);
    tri_origins.allocate(n_tris);
    tf::parallel_for_each(
        tf::make_sequence_range(n_tris), [&](std::size_t e) {
          const auto tag = tags[Index(e)];
          const auto &tr = tris[Index(e)];
          for (int c = 0; c < 3; ++c)
            tri_data[3 * e + std::size_t(c)] =
                map_data.map_vertex(tag, tr[std::size_t(c)]);
          tri_tags[e] = tag;
          tri_origins[e] = descs[std::size_t(slots[Index(e)])].object;
        });
  }

  auto triangles = tf::make_blocked_range<3>(tf::make_range(tri_data));

  // Output faces: each tag's uncut faces then the cut triangles. How the
  // uncut views are gathered is the policy's business — one range over a
  // homogeneous set of operands, two separate ones for a pair.
  auto faces = policy.template concatenated_output_faces<Index>(map_data,
                                                                triangles);

  // The created half is the unified table: intersection points followed
  // by any refinement-added ones.
  auto total_pts =
      map_data.total_original_points + map_data.total_created_points;
  tf::points_buffer<RealOut, 3> pts_buf;
  pts_buf.allocate(total_pts);

  {
    // the reader outlives the wait: the emission's tasks read it
    const auto reader = arrangement.lattice().reader(apply_to_polygons);
    tbb::task_group tg;
    tf::arrangement::emit_arrangement_points<RealOut>(
        tg, n_meshes, apply_to_polygons, map_data, created_pts, reader,
        pts_buf);
    tg.wait();
  }

  auto total_faces = static_cast<Index>(faces.size());
  tf::buffer<Index> tag_labels;
  tf::buffer<Index> face_labels;
  // A single operand tags every face 0, and no reader of a one-form
  // arrangement carries a tag axis, so that buffer is not built.
  constexpr bool tagged = Policy::static_n_tags != 1;
  if constexpr (tagged)
    tag_labels.allocate(total_faces);
  face_labels.allocate(total_faces);

  {
    [[maybe_unused]] auto tag_uncut =
        tf::make_offset_block_range(map_data.original_face_offsets, tag_labels);
    auto face_uncut = tf::make_offset_block_range(
        map_data.original_face_offsets, face_labels);
    tbb::task_group tg;
    for (Index t = 0; t < n_meshes; ++t) {
      tg.run([&, t] {
        constexpr bool tagged = Policy::static_n_tags != 1;
        if constexpr (tagged)
          tf::parallel_fill(tag_uncut[t], t);
        tf::parallel_copy(
            tf::make_range(map_data.original_face_ids[std::size_t(t)]),
            face_uncut[t]);
      });
    }
    tg.run([&] {
      constexpr bool tagged = Policy::static_n_tags != 1;
      auto tri_off = map_data.total_original_faces;
      if constexpr (tagged)
        tf::parallel_copy(tri_tags, tf::drop(tag_labels, tri_off));
      tf::parallel_copy(tri_origins, tf::drop(face_labels, tri_off));
    });
    tg.wait();
  }

  return std::make_tuple(
      tf::make_polygons_buffer(std::move(faces), std::move(pts_buf)),
      std::move(tag_labels), std::move(face_labels), std::move(map_data));
}

} // namespace tf::arrangement

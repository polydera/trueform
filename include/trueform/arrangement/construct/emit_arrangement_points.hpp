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
#include "../../core/frame_of.hpp"
#include "../../core/points.hpp"
#include "../../core/points_buffer.hpp"
#include "../../core/range.hpp"
#include "../../core/transformed.hpp"
#include "../../core/views/drop.hpp"
#include "../../core/views/indirect_range.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/offset_block_range.hpp"
#include "./arrangement_map_data.hpp"

#include "tbb/task_group.h"

#include <cstddef>
#include <type_traits>

namespace tf::arrangement {

/// The output point table of an arrangement emission: each tag's used
/// original vertices into its own slice, then the created table behind
/// them. `pts_buf` must already hold
/// `total_original_points + total_created_points` slots.
///
/// BOTH HALVES COME OUT OF ONE READER. A created point is derived from
/// where the originals stand, so an emission taking the created half
/// from the pipeline and the original half from the input's own float
/// would open every cut seam by up to the tolerance. At tolerance zero
/// the reader's real answer IS the input's coordinate: an export is a
/// view of the input, not a round trip through the lattice.
///
/// The tasks are scheduled on the caller's group and run until the caller
/// waits: the reader is borrowed and must outlive that wait, and every
/// view taken off a form is captured by value.
template <typename RealOut, typename Index, typename ApplyToForm,
          typename CreatedPoints, typename Reader>
void emit_arrangement_points(tbb::task_group &tg, Index n_tags,
                             const ApplyToForm &apply_to_form,
                             const arrangement_point_map_data<Index> &map_data,
                             const CreatedPoints &created_pts,
                             const Reader &reader,
                             tf::points_buffer<RealOut, 3> &pts_buf) {
  auto pts_range =
      tf::make_offset_block_range(map_data.original_offsets, pts_buf);
  for (Index t = 0; t < n_tags; ++t) {
    tg.run([&, t, pts_range] {
      apply_to_form(t, [&](const auto &form) {
        auto frame = tf::frame_of(form);
        auto points = form.points();
        const auto base = reader.vertex_offsets[std::size_t(t)];
        if constexpr (std::is_integral_v<RealOut>) {
          tf::parallel_copy(
              tf::make_points(tf::make_mapped_range(
                  map_data.original_ids[t],
                  [&reader, points, frame, base](Index id) {
                    return reader.point_in(points, frame, base, id);
                  })),
              pts_range[t]);
        } else {
          tf::parallel_copy(
              tf::make_points(tf::make_mapped_range(
                  map_data.original_ids[t],
                  [&reader, points, frame, base](Index id) {
                    return reader.real_point_in(points, frame, base, id);
                  })),
              pts_range[t]);
        }
      });
    });
  }
  tg.run([&] {
    auto created = tf::make_indirect_range(tf::make_range(map_data.created_ids),
                                           tf::make_range(created_pts));
    if constexpr (std::is_integral_v<RealOut>) {
      tf::parallel_copy(tf::make_points(created),
                        tf::drop(pts_buf, map_data.total_original_points));
    } else {
      tf::parallel_copy(
          tf::make_points(tf::make_mapped_range(
              created,
              [&reader](auto pt) { return reader.converter.deconvert(pt); })),
          tf::drop(pts_buf, map_data.total_original_points));
    }
  });
}

} // namespace tf::arrangement

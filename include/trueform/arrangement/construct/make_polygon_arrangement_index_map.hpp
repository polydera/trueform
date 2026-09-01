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

#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../polygon_arrangement_index_map.hpp"
#include "./arrangement_map_data.hpp"

#include <cstddef>

namespace tf::arrangement {

/// Finalize a (discarded) arrangement_map_data at one operand plus the
/// construct's face_labels into the public single-mesh index map. With a
/// single mesh `point_offsets[0]` and `original_offsets[0]` are both 0, so
/// `original_map` is already global and `point_f` is a pure move; the point
/// inverse is `original_ids[0]` padded with an `end`-sentinel tail for the
/// created points.
template <typename Index>
auto make_polygon_arrangement_index_map(arrangement_map_data<Index> &&d,
                                        tf::buffer<Index> &&face_labels,
                                        Index n_original_faces,
                                        Index n_output_points)
    -> tf::polygon_arrangement_index_map<Index> {
  tf::polygon_arrangement_index_map<Index> out;
  out.n_original_points = d.total_original_points;
  out.n_original_faces = n_original_faces;
  out.n_output_points = n_output_points;
  out.point_f = std::move(d.original_map);
  out.face_labels = std::move(face_labels);

  const Index in_end = static_cast<Index>(out.point_f.size());
  tf::parallel_for_each(
      out.point_f,
      [in_end, n_output_points](Index &v) {
        if (v == in_end)
          v = n_output_points;
      },
      tf::checked);

  out.point_labels.allocate(static_cast<std::size_t>(n_output_points));
  tf::parallel_fill(out.point_labels, n_output_points);
  const auto &ids = d.original_ids[0];
  tf::parallel_for_each(
      tf::make_sequence_range(Index(0), out.n_original_points),
      [&](Index o) { out.point_labels[o] = ids[std::size_t(o)]; }, tf::checked);
  return out;
}

} // namespace tf::arrangement

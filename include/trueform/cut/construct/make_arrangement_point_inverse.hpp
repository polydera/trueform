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
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./arrangement_map_data.hpp"

namespace tf::cut {

/// Fill the output-point inverse of an arrangement point map: for each global
/// output point, which input form (`tag`) and which input point (`label`) it
/// came from. Kept originals map to their real `(t, ids[k])`; created points
/// keep the `end` sentinels — the tag axis ends at `n_tags` (= `d.n_meshes`),
/// the point-id axis at `n_output_points`. Shared by the mesh arrangement
/// index map and the CSG domains index map, which both work off the same
/// `arrangement_point_map_data` base.
template <typename Index>
auto make_arrangement_point_inverse(const arrangement_point_map_data<Index> &d,
                                    Index n_output_points,
                                    tf::buffer<Index> &tag,
                                    tf::buffer<Index> &label) -> void {
  const Index n = d.n_meshes;
  tag.allocate(static_cast<std::size_t>(n_output_points));
  label.allocate(static_cast<std::size_t>(n_output_points));
  tf::parallel_fill(tag, n);
  tf::parallel_fill(label, n_output_points);
  for (Index t = 0; t < n; ++t) {
    const Index base = d.original_offsets[t];
    const auto &ids = d.original_ids[t];
    tf::parallel_for_each(
        tf::make_sequence_range(Index(0), static_cast<Index>(ids.size())),
        [&](Index k) {
          tag[static_cast<std::size_t>(base + k)] = t;
          label[static_cast<std::size_t>(base + k)] = ids[k];
        },
        tf::checked);
  }
}

} // namespace tf::cut

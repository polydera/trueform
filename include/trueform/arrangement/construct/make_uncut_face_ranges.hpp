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
#include "../../core/blocked_buffer.hpp"
#include "./arrangement_map_data.hpp"
#include <cstddef>

namespace tf::arrangement {

/// @brief Per-tag `[begin, end)` of the faces emitted uncut, for the
///        arrangement emission order: every tag's uncut faces first,
///        grouped by tag, then all cut triangles.
///
/// That grouping is exactly `original_face_offsets`, so this reads the
/// ranges off the map data rather than counting anything again.
template <typename Index>
auto make_uncut_face_ranges(
    const tf::arrangement::arrangement_map_data<Index> &d)
    -> tf::blocked_buffer<Index, 2> {
  tf::blocked_buffer<Index, 2> out;
  out.allocate(static_cast<std::size_t>(d.n_meshes));
  for (Index t = 0; t < d.n_meshes; ++t) {
    out[t][0] = d.original_face_offsets[t];
    out[t][1] = d.original_face_offsets[t + 1];
  }
  return out;
}

} // namespace tf::arrangement

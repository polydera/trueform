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

#include <algorithm>
#include <cstddef>
#include <iterator>

namespace tf::exact {

/// The tag a flat original vertex id belongs to: `vertex_offsets` is the
/// per-tag prefix, so the tag is the last offset not above the id. An id
/// outside the prefix answers with the nearest tag.
template <typename Index, typename VertexOffsets>
auto tag_of_flat_vertex(const VertexOffsets &vertex_offsets, Index id)
    -> Index {
  const auto at =
      std::upper_bound(vertex_offsets.begin(), vertex_offsets.end(), id);
  const auto tag = std::distance(vertex_offsets.begin(), at) - 1;
  const auto last = std::ptrdiff_t(vertex_offsets.size()) - 2;
  return Index(tag < 0 ? 0 : (tag > last ? last : tag));
}

} // namespace tf::exact

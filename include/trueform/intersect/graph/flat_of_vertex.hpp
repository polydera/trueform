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

#include <cstddef>
#include <cstdint>

namespace tf::intersect::graph {

/// The flat identity of one vertex: `vertex_offsets` is the per-tag prefix, so
/// an original vertex lifts by its own tag's offset and a created vertex lifts
/// past the whole original space, whose extent is the prefix's last entry.
/// The inverse of @ref tf::exact::tag_of_flat_vertex.
template <typename Index, typename VertexOffsets>
auto flat_of_vertex(const VertexOffsets &vertex_offsets, std::int16_t tag,
                    Index id) -> Index {
  return tag < 0 ? Index(vertex_offsets[vertex_offsets.size() - 1]) + id
                 : Index(vertex_offsets[std::size_t(tag)]) + id;
}

} // namespace tf::intersect::graph

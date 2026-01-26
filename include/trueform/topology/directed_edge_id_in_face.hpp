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

namespace tf {

/// @ingroup topology_types
/// @brief Finds the index of a directed edge within a face.
///
/// Searches for the edge (v0 → v1) in the face, where the edge starts at
/// vertex v0 and ends at vertex v1 in the face's winding order.
///
/// @tparam Index The integer type for indices.
/// @tparam Range The face range type.
/// @param v0 The start vertex of the directed edge.
/// @param v1 The end vertex of the directed edge.
/// @param face The face to search in.
/// @return The edge index (0 to size-1) if found, or size if not found.
template <typename Index, typename Range>
auto directed_edge_id_in_face(const Index &v0, const Index &v1,
                              const Range &face) {
  Index size = face.size();
  Index prev = size - 1;
  for (Index i = 0; i < size; prev = i++) {
    if (char(face[prev] == v0) & char(face[i] == v1))
      return prev;
  }
  return size;
}
} // namespace tf

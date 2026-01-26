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

#include <cstddef>
namespace tf {

/// @ingroup topology_types
/// @brief Finds the index of an undirected edge within a face.
///
/// Searches for the edge connecting v0 and v1 in the face, regardless of
/// direction. Matches both (v0 → v1) and (v1 → v0) orderings.
///
/// @tparam Index The integer type for vertex indices.
/// @tparam Range The face range type.
/// @param v0 One endpoint of the edge.
/// @param v1 The other endpoint of the edge.
/// @param face The face to search in.
/// @return The edge index (0 to size-1) if found, or size if not found.
/// @see tf::directed_edge_id_in_face() for direction-sensitive search.
template <typename Index, typename Range>
auto edge_id_in_face(const Index &v0, const Index &v1, const Range &face) {
  std::size_t size = face.size();
  std::size_t prev = size - 1;
  for (std::size_t i = 0; i < size; prev = i++) {
    if ((char(face[prev] == v0) & char(face[i] == v1)) |
        (char(face[prev] == v1) & char(face[i] == v0)))
      return prev;
  }
  return size;
}
} // namespace tf

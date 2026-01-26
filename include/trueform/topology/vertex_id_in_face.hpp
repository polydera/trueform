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
/// @brief Finds the index of a vertex within a face.
///
/// Searches for vertex v in the face's vertex list.
///
/// @tparam Index The integer type for indices.
/// @tparam Range The face range type.
/// @param v The vertex to search for.
/// @param face The face to search in.
/// @return The vertex index (0 to size-1) if found, or size if not found.
template <typename Index, typename Range>
auto vertex_id_in_face(const Index &v, const Range &face) {
  Index size = face.size();
  for (Index i = 0; i < size; i++) {
    if (face[i] == v)
      return i;
  }
  return size;
}
} // namespace tf

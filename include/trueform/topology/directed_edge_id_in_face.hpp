/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf {
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

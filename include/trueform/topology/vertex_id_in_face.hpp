/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf {
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

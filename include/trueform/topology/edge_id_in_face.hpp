/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf {
template <typename Index, typename Range>
auto edge_id_in_face(const Index &v0, const Index &v1, const Range &face) {
  Index size = face.size();
  Index prev = size - 1;
  for (Index i = 0; i < size; prev = i++) {
    if ((char(face[prev] == v0) & char(face[i] == v1)) |
        (char(face[prev] == v1) & char(face[i] == v0)))
      return prev;
  }
  return size;
}
} // namespace tf

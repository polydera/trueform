/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via info@polydera.com.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

#include <cstddef>
namespace tf {
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

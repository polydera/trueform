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
#include "../../core/faces.hpp"
#include <cstddef>

namespace tf::fill {

/// Whether a face winds the undirected edge `a b`, in either direction.
template <typename Policy, typename Index>
auto hole_face_carries_edge(const tf::faces<Policy> &faces, Index face,
                            Index a, Index b) -> bool {
  const auto corners = faces[std::size_t(face)];
  const Index n = Index(corners.size());
  for (Index i = 0; i < n; ++i) {
    const Index j = Index(i + 1 == n ? 0 : i + 1);
    if ((corners[std::size_t(i)] == a && corners[std::size_t(j)] == b) ||
        (corners[std::size_t(i)] == b && corners[std::size_t(j)] == a))
      return true;
  }
  return false;
}

/// Whether a face names `vertex` among its corners.
template <typename Policy, typename Index>
auto hole_face_has_corner(const tf::faces<Policy> &faces, Index face,
                          Index vertex) -> bool {
  for (auto corner : faces[std::size_t(face)])
    if (corner == vertex)
      return true;
  return false;
}

} // namespace tf::fill

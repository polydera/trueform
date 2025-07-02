/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Distributed under the Boost Software License, Version 1.0.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf {
template <typename Index> struct manifold_edge_peer {
  static constexpr Index boundary = -1;
  static constexpr Index non_manifold = -2;

  Index face_peer;

  auto is_simple() const -> bool { return face_peer >= 0; }

  auto is_boundary() const -> bool { return face_peer == boundary; }

  auto is_manifold() const -> bool { return face_peer > non_manifold; }
};
} // namespace tf

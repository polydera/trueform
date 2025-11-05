/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial License 1.0.0.
 * Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once

namespace tf {
template <typename Index> struct manifold_edge_peer {
  static constexpr Index boundary = -1;
  static constexpr Index non_manifold = -2;
  static constexpr Index non_manifold_representative = -3;

  Index face_peer;

  auto is_simple() const -> bool { return face_peer >= 0; }

  auto is_boundary() const -> bool { return face_peer == boundary; }

  auto is_manifold() const -> bool { return face_peer > non_manifold; }

  auto is_representative(Index from_polygon) const -> bool {
    return face_peer == boundary || face_peer == non_manifold_representative ||
           (face_peer >= 0 && from_polygon < face_peer);
  }
};
} // namespace tf

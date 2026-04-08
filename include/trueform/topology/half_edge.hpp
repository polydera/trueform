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

#include <type_traits>

namespace tf {

/// @ingroup topology_connectivity
/// @brief Represents a half-edge in a half-edge data structure.
///
/// Each half-edge stores its face, the vertex it originates from,
/// and the handle of the next half-edge. Sentinel values encode boundary,
/// non-manifold, and orientation fault states.
///
/// Opposite half-edges are stored at adjacent indices (index XOR 1).
///
/// @tparam Index The signed integer type for indices.
template <typename Index> struct half_edge {
  static_assert(std::is_signed_v<Index>, "Index must be signed");

  static constexpr Index boundary = -1;
  static constexpr Index non_manifold = -2;
  static constexpr Index orientation_fault = -3;
  static constexpr Index removed = -4;
  static constexpr Index no_id = -1;

  Index face;
  Index vertex;
  Index next;
  Index prev;

  /// @brief Returns true if this half-edge has a valid next pointer.
  auto has_next() const -> bool { return next >= 0; }

  /// @brief Returns true if this half-edge belongs to a face.
  auto is_simple() const -> bool { return face >= 0; }

  /// @brief Returns true if this is a boundary half-edge (no adjacent face).
  auto is_boundary() const -> bool { return face == boundary; }

  /// @brief Returns true if this half-edge is manifold (simple or boundary).
  auto is_manifold() const -> bool { return face > non_manifold; }

  /// @brief Returns true if this half-edge has a faulty orientation.
  auto is_faulty_oriented() const -> bool {
    return face == orientation_fault;
  }

  /// @brief Returns true if this half-edge has been removed.
  auto is_removed() const -> bool { return face == removed; }
};

} // namespace tf

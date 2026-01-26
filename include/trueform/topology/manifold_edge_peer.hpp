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
/// @brief Stores peer face information for a directed edge.
///
/// Used by @ref tf::manifold_edge_link to store connectivity information
/// for each directed edge. The `face_peer` field indicates:
/// - A non-negative value: The index of the adjacent face sharing this edge
/// - `boundary`: The edge is on the mesh boundary (no peer face)
/// - `non_manifold`: The edge is shared by more than 2 faces
/// - `non_manifold_representative`: The representative edge among non-manifold edges
///
/// @tparam Index The integer type for face indices.
template <typename Index> struct manifold_edge_peer {
  /// @brief Sentinel value indicating a boundary edge.
  static constexpr Index boundary = -1;
  /// @brief Sentinel value indicating a non-manifold edge.
  static constexpr Index non_manifold = -2;
  /// @brief Sentinel value for the representative non-manifold edge.
  static constexpr Index non_manifold_representative = -3;

  /// @brief The peer face index, or a sentinel value.
  Index face_peer;

  /// @brief Check if this is a simple manifold edge (exactly 2 faces).
  /// @return `true` if the edge has a valid peer face.
  auto is_simple() const -> bool { return face_peer >= 0; }

  /// @brief Check if this is a boundary edge.
  /// @return `true` if the edge belongs to only one face.
  auto is_boundary() const -> bool { return face_peer == boundary; }

  /// @brief Check if this is a manifold edge.
  /// @return `true` if the edge is boundary or simple (not non-manifold).
  auto is_manifold() const -> bool { return face_peer > non_manifold; }

  /// @brief Check if this edge is the representative for its equivalence class.
  ///
  /// For boundary and non-manifold representative edges, always returns `true`.
  /// For simple manifold edges, returns `true` if `from_polygon < face_peer`,
  /// ensuring exactly one of the two faces is the representative.
  ///
  /// @param from_polygon The face index this edge belongs to.
  /// @return `true` if this is the representative edge.
  auto is_representative(Index from_polygon) const -> bool {
    return face_peer == boundary || face_peer == non_manifold_representative ||
           (face_peer >= 0 && from_polygon < face_peer);
  }
};
} // namespace tf

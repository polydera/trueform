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
/// @brief The tet across one facet of a tet, or the facet's sentinel.
///
/// The @ref tf::manifold_edge_peer one dimension up, with its sentinels:
/// `boundary` where no other tet holds the facet, `non_manifold` where more
/// than one does, `non_manifold_representative` on the smallest such tet.
///
/// @tparam Index The integer type for tet indices.
template <typename Index> struct manifold_facet_peer {
  static constexpr Index boundary = -1;
  static constexpr Index non_manifold = -2;
  static constexpr Index non_manifold_representative = -3;

  Index tet_peer;

  /// @brief Exactly two tets hold the facet.
  auto is_simple() const -> bool { return tet_peer >= 0; }

  auto is_boundary() const -> bool { return tet_peer == boundary; }

  /// @brief At most two tets hold the facet.
  auto is_manifold() const -> bool { return tet_peer > non_manifold; }

  /// @brief Exactly one holder of each facet answers true.
  auto is_representative(Index from_tet) const -> bool {
    return tet_peer == boundary || tet_peer == non_manifold_representative ||
           (tet_peer >= 0 && from_tet < tet_peer);
  }
};
} // namespace tf

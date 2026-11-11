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
#include "../../../core/buffer.hpp"
#include "../../../core/config/verify.hpp"
#include "./state_tet_facet_protection.hpp"
#include "./tet_cavity_links.hpp"
#include "./tet_facet_is_protected.hpp"
#include "./tet_state.hpp"
#include "./write_tet_topology.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Write and link a cavity star into assigned cell rows, the boundary as
/// `links` states it: cell `i` stands on facet `i`, and across each of its
/// side walls the cell of the facet sharing that edge. Boundary facet rows
/// and output rows have one writer, which states each row's liveness; the
/// caller owns scratch. A star cell's base facet keeps the protection, and the
/// owner, its boundary facet held.
template <typename Owner, typename Boundary, typename Index>
auto write_tet_cavity(Owner &owner, typename Owner::index_type apex,
                      const tf::buffer<Index> &rows, const Boundary &boundary,
                      const tet_cavity_links<Index> &links) -> void {
  const std::size_t walls = boundary.size();
  if constexpr (tf::core::verification)
    tf::core::verify(links.n_facets == walls,
                     "the links state the cavity boundary");
  // A star cell of a stock complex, with no protected lane and nothing
  // recorded, standing on a finite facet is the facet's corners as it winds
  // them with the new site last, and its walls stand in the facet's own edge
  // order.
  const bool stock = owner._protected_slots.size() == 0 &&
                     !owner._record_site_cells && !owner._record_written;
  for (std::size_t i = 0; i < walls; ++i) {
    const std::size_t ticket = std::size_t(boundary[i]);
    const std::size_t side = ticket / 4;
    const std::size_t slot = ticket % 4;
    const auto &facet = links.facets[i];
    const Index tet = rows[i];
    if (stock && facet[0] != Owner::infinite && facet[1] != Owner::infinite &&
        facet[2] != Owner::infinite) {
      auto row = owner._corners[std::size_t(tet)];
      auto link = owner._neighbors[std::size_t(tet)];
      for (std::size_t e = 0; e < 3; ++e) {
        const Index partner = links.across[3 * i + e];
        if constexpr (tf::core::verification)
          tf::core::verify(partner >= 0,
                           "every side wall of a star cell has a partner");
        row[e] = facet[e];
        link[e] = rows[std::size_t(partner) / 3];
      }
      row[3] = apex;
      link[3] = Index(side);
      owner._neighbors[side][slot] = tet;
      owner._states[std::size_t(tet)] = tet_state::live;
      continue;
    }
    const std::size_t base =
        write_tet_topology(owner, tet, facet[0], facet[1], facet[2], apex);
    owner._neighbors[std::size_t(tet)][base] = Index(side);
    owner._neighbors[side][slot] = tet;
    owner._states[std::size_t(tet)] =
        owner._corners[std::size_t(tet)][3] == Owner::infinite
            ? tet_state::infinite
            : tet_state::live;
    if (tet_facet_is_protected(owner, Index(side), slot))
      state_tet_facet_protection(owner, Index(side), slot, true,
                                 owner._owners[4 * side + slot]);
    const auto corners = owner._corners[std::size_t(tet)];
    for (std::size_t e = 0; e < 3; ++e) {
      const Index partner = links.across[3 * i + e];
      if constexpr (tf::core::verification)
        tf::core::verify(partner >= 0,
                         "every side wall of a star cell has a partner");
      const std::size_t wall = std::size_t(corners[1] == facet[e]) +
                               2 * std::size_t(corners[2] == facet[e]) +
                               3 * std::size_t(corners[3] == facet[e]);
      owner._neighbors[std::size_t(tet)][wall] = rows[std::size_t(partner) / 3];
    }
  }
}

} // namespace tf::topology::cdt::dt3

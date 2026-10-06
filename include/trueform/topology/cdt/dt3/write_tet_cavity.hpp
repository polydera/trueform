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
#include "./advance_tet_cavity_edges.hpp"
#include "./link_tet_cavity_edge.hpp"
#include "./link_tet_sides.hpp"
#include "./tet_cavity_edge_table.hpp"
#include "./tet_facets.hpp"
#include "./tet_side.hpp"
#include "./write_tet_topology.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// Write and link a cavity star into assigned cell rows. Boundary facet rows
/// and output rows have one writer; the caller owns scratch and publishes
/// states.
template <typename Owner, typename Boundary, typename Rows, typename Index>
auto write_tet_cavity(Owner &owner, typename Owner::index_type apex,
                      const Rows &rows, const Boundary &boundary,
                      tf::buffer<tet_side<typename Owner::index_type>> &sides,
                      tet_cavity_edge_table<Index> &edges) -> void {
  const std::size_t walls = boundary.size();
  const bool small = walls <= tet_cavity_edge_table<Index>::max_walls;
  if (small)
    advance_tet_cavity_edges(edges);
  else
    sides.allocate(3 * walls);

  for (std::size_t i = 0; i < walls; ++i) {
    const std::size_t ticket = std::size_t(boundary[i]);
    const std::size_t side = ticket / 4;
    const std::size_t slot = ticket % 4;
    const auto facet = tet_facet(owner._corners[side], slot);
    const Index tet = rows[i];
    const std::size_t base =
        write_tet_topology(owner, tet, facet[0], facet[1], facet[2], apex);
    owner._neighbors[std::size_t(tet)][base] = Index(side);
    owner._neighbors[side][slot] = tet;

    const auto corners = owner._corners[std::size_t(tet)];
    std::size_t taken = 0;
    for (std::size_t wall = 0; wall < 4; ++wall) {
      if (wall == base)
        continue;
      std::array<Index, 2> edge{};
      std::size_t end = 0;
      for (std::size_t corner = 0; corner < 4; ++corner)
        if (corner != base && corner != wall)
          edge[end++] = corners[corner];
      if (edge[1] < edge[0])
        std::swap(edge[0], edge[1]);
      if (small)
        link_tet_cavity_edge(owner, edges, edge[0], edge[1], tet, int(wall));
      else
        sides[3 * i + taken++] = {edge[0], edge[1], tet, Index(wall)};
    }
  }

  if (small)
    return;
  link_tet_sides(owner, sides);
}

} // namespace tf::topology::cdt::dt3

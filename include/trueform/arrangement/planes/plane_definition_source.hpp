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

#include "../../core/buffer.hpp"
#include "../../intersect/graph/plane_tables.hpp"

#include <cstddef>

namespace tf::arrangement {

/// One plane's block: the table that holds it, its index there, and whether
/// that table is the immutable world's.
template <typename Index, typename Int> struct plane_definition_source {
  const tf::intersect::graph::plane_tables<Index, Int> &tables;
  Index block;
  bool immutable;
};

/// CORE. Whether one plane still reads the world tier. The ticket is the whole
/// fact, and a tier holding no block at all states it for every plane.
template <typename Index>
auto plane_reads_world_tier(const tf::buffer<Index> &plane_ticket, Index plane)
    -> bool {
  return plane_ticket.size() == 0 ||
         plane_ticket[std::size_t(plane)] == Index(-1);
}

/// CORE. The plane space a ticket answers in: its own extent once this
/// arrangement states one, and the world's carriers before that.
template <typename Index, typename Int>
auto plane_ticket_space(
    const tf::intersect::graph::plane_tables<Index, Int> &world_tables,
    const tf::buffer<Index> &plane_ticket) -> std::size_t {
  return plane_ticket.size() == 0 ? world_tables.edges().size()
                                  : plane_ticket.size();
}

/// CORE. The tier one plane's BLOCK lives in — the world's while its ticket is
/// `-1`, this arrangement's otherwise. The rows inside it name their own tier:
/// AN UNCHANGED GROUP STAYS THE WORLD'S, VERBATIM, FOR BOTH CARRIERS, so a
/// block this arrangement holds still names the world's own rows for every
/// group no wave changed, and @ref tf::arrangement::plane_tier_definitions is
/// what reads them.
template <typename Index, typename Int>
auto find_plane_definition_source(
    const tf::intersect::graph::plane_tables<Index, Int> &world_tables,
    const tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    const tf::buffer<Index> &plane_ticket, Index plane)
    -> plane_definition_source<Index, Int> {
  return plane_reads_world_tier(plane_ticket, plane)
             ? plane_definition_source<Index, Int>{world_tables, plane, true}
             : plane_definition_source<Index, Int>{
                   local_tables, plane_ticket[std::size_t(plane)], false};
}

template <typename Index, typename Int, typename World>
auto find_plane_definition_source(
    const World &world,
    const tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    const tf::buffer<Index> &plane_ticket, Index plane)
    -> plane_definition_source<Index, Int> {
  return find_plane_definition_source(world.tables(), local_tables,
                                      plane_ticket, plane);
}

/// CORE. One plane's constraint count, read off the tier that answers it.
template <typename Index, typename Int, typename World>
auto plane_definition_edge_count(
    const World &world,
    const tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    const tf::buffer<Index> &plane_ticket, Index plane) -> Index {
  const auto source =
      find_plane_definition_source(world, local_tables, plane_ticket, plane);
  return Index(source.tables.plane_edges(source.block).size());
}

} // namespace tf::arrangement

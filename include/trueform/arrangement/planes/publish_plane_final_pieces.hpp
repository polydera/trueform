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
#include "../../core/offset_block_buffer.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_tables.hpp"
#include "./finalize_plane_piece_tickets.hpp"
#include "./make_plane_final_piece_definitions.hpp"
#include "./plane_tier_definitions.hpp"

#include <array>
#include <cstddef>

namespace tf::arrangement {

/// WAVE. The boundary lift: every constrained slot published in the final
/// piece space, and the complete definition span of every piece this tier
/// owns.
template <typename Index, typename Int, typename World, typename VertexOffsets>
auto publish_plane_final_pieces(
    const World &world,
    const tf::intersect::graph::plane_tables<Index, Int> &local_tables,
    const tf::buffer<Index> &group_router,
    const tf::buffer<Index> &plane_ticket,
    const VertexOffsets &vertex_offsets, Index n_flat,
    Index immutable_canon_extent,
    tf::offset_block_buffer<Index,
                            tf::intersect::graph::plane_edge_def<Index>>
        &final_piece_definitions,
    const tf::buffer<std::array<Index, 3>> &triangles,
    tf::buffer<std::array<Index, 3>> &slot_parents,
    const tf::buffer<Index> &plane_offsets) -> bool {
  return make_plane_final_piece_definitions(world, local_tables,
                                            immutable_canon_extent,
                                            final_piece_definitions) &&
         finalize_plane_piece_tickets(
             world, local_tables,
             make_plane_tier_definitions(world.tables(), local_tables, true),
             group_router, plane_ticket, vertex_offsets, n_flat,
             immutable_canon_extent, triangles, slot_parents, plane_offsets);
}

} // namespace tf::arrangement

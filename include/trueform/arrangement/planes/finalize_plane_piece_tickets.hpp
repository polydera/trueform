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

#include "../../core/algorithm/parallel_contains.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "./plane_group_router.hpp"
#include "./plane_piece_key.hpp"
#include "./plane_tier_definitions.hpp"

#include <cassert>
#include <cstddef>
#include <limits>

namespace tf::arrangement {

/// Publish every constrained triangle slot in the final generation's piece
/// space. Immutable groups are the prefix; PA-owned groups are the suffix.
///
/// AN UNCHANGED GROUP STAYS THE WORLD'S, VERBATIM, FOR BOTH CARRIERS, so a
/// carrier this arrangement holds and one still reading the world publish the
/// SAME prefix ticket for it — nothing is resolved on an untouched carrier,
/// because nothing about its pieces moved. A carrier whose block the wave did
/// rewrite reads its slots off that block again, by key, since a split or a
/// fusion is exactly what its standing tickets no longer name.
template <typename Index, typename Immutable, typename Current,
          typename Tier, typename PlaneTickets, typename VertexOffsets,
          typename Triangles, typename SlotParents, typename PlaneOffsets>
auto finalize_plane_piece_tickets(
    const Immutable &immutable, const Current &current, const Tier &tier,
    const tf::buffer<Index> &router, const PlaneTickets &plane_ticket,
    const VertexOffsets &vertex_offsets, Index n_flat, Index immutable_extent,
    const Triangles &triangles, SlotParents &slot_parents,
    const PlaneOffsets &plane_offsets) -> bool {
  if (plane_ticket.size() == 0)
    return true;
  if (plane_ticket.size() == std::numeric_limits<std::size_t>::max() ||
      plane_ticket.size() >
          std::size_t(std::numeric_limits<Index>::max()) ||
      immutable_extent < Index(0) || current.n_canon() < Index(0) ||
      current.n_canon() >
          std::numeric_limits<Index>::max() - immutable_extent ||
      router.size() != std::size_t(immutable_extent) ||
      plane_offsets.size() != plane_ticket.size() + 1 ||
      slot_parents.size() != triangles.size())
    return false;
  const auto final_offset = plane_offsets[plane_offsets.size() - 1];
  if (plane_offsets[0] != Index(0) || final_offset < Index(0) ||
      std::size_t(final_offset) != triangles.size())
    return false;
  if (tf::parallel_contains(
          tf::make_sequence_range(plane_ticket.size()),
          [&plane_offsets](std::size_t plane) {
            return plane_offsets[plane] < Index(0) ||
                   plane_offsets[plane] > plane_offsets[plane + 1];
          },
          tf::checked))
    return false;

  tf::buffer<char> valid;
  valid.allocate_and_initialize(plane_ticket.size(), char(1));
  tf::parallel_for_each(
      tf::make_sequence_range(Index(plane_ticket.size())), [&](Index plane) {
        const auto current_plane = plane_ticket[std::size_t(plane)];
        const auto finalize_active = [&](const auto &block) {
          for (auto triangle = plane_offsets[std::size_t(plane)];
               triangle < plane_offsets[std::size_t(plane) + 1]; ++triangle) {
            const auto &corners = triangles[std::size_t(triangle)];
            auto &tickets = slot_parents[std::size_t(triangle)];
            for (std::size_t slot = 0; slot < 3; ++slot) {
              const auto ticket = find_plane_piece_ticket<Index>(
                  tier, block,
                  plane_piece_key(corners[slot],
                                  corners[(slot + std::size_t(1)) % 3], n_flat,
                                  vertex_offsets));
              if (ticket == Index(-1) && tickets[slot] != Index(-1))
                valid[std::size_t(plane)] = char(0);
              tickets[slot] = ticket;
            }
          }
        };
        // THE OWNERSHIP LAW: an immutable plane names no group this
        // arrangement took — taking a group took every carrier of it — so an
        // untouched carrier's slots already carry their final immutable ids,
        // and there is nothing to resolve
        const auto finalize_inactive = [&]() {
#ifndef NDEBUG
          for (auto triangle = plane_offsets[std::size_t(plane)];
               triangle < plane_offsets[std::size_t(plane) + 1]; ++triangle) {
            const auto &tickets = slot_parents[std::size_t(triangle)];
            // an untouched carrier names an immutable group the router states
            // and still leaves to the world — never a retired root, whose
            // staging closure took every carrier of it
            for (std::size_t slot = 0; slot < 3; ++slot)
              assert(tickets[slot] == Index(-1) ||
                     (std::size_t(tickets[slot]) < router.size() &&
                      resolve_plane_group_router(router, tickets[slot],
                                                 Index(-1)) == Index(-1)));
          }
#endif
        };
        if (current_plane == Index(-1)) {
          if (plane >= immutable.n_planes()) {
            if (plane_offsets[std::size_t(plane)] !=
                plane_offsets[std::size_t(plane) + 1])
              valid[std::size_t(plane)] = char(0);
          } else {
            finalize_inactive();
          }
        } else if (current_plane < Index(0) ||
                   current_plane >= current.n_planes()) {
          valid[std::size_t(plane)] = char(0);
        } else {
          finalize_active(current.plane_edges(current_plane));
        }
      });
  return !tf::parallel_contains(
      tf::make_range(valid), [](char value) { return value == char(0); },
      tf::checked);
}

} // namespace tf::arrangement

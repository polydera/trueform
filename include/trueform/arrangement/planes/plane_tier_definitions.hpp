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
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../intersect/graph/plane_edge_def.hpp"
#include "../../intersect/graph/plane_tables.hpp"

#include <cstddef>

namespace tf::arrangement {

/// CORE. Whether every ticket names a tier that can answer it: a block this
/// arrangement's local tier holds, or the world tier by `-1` — which a plane
/// past the world's own carriers has no right to name.
template <typename Index>
auto plane_ticket_well_formed(const tf::buffer<Index> &plane_ticket,
                              Index n_world_planes, Index n_local_blocks)
    -> bool {
  return !tf::parallel_contains(
      tf::make_sequence_range(plane_ticket.size()),
      [&plane_ticket, n_world_planes, n_local_blocks](std::size_t plane) {
        const auto block = plane_ticket[plane];
        return block < Index(-1) || block >= n_local_blocks ||
               (plane >= std::size_t(n_world_planes) && block == Index(-1));
      },
      tf::checked);
}

/// THE TWO TIERS A PLANE BLOCK READS.
///
/// A block of the world tier names the world's own definition rows, and
/// nothing else. A block THIS ARRANGEMENT holds names its own rows, and a
/// NEGATIVE row is the world's, at `-1 - row`: AN UNCHANGED GROUP STAYS THE
/// WORLD'S, VERBATIM, FOR BOTH CARRIERS, so a carrier the wave ported for
/// another group's sake keeps naming the world's rows for it, and the carrier
/// the wave never touched reads exactly the same rows and publishes exactly
/// the same ticket.
///
/// One reader answers a block of either tier: which one it is, is the block's
/// own fact, stated once when the reader is made.
template <typename Index, typename Int> class plane_tier_definitions {
public:
  using def_t = tf::intersect::graph::plane_edge_def<Index>;

  plane_tier_definitions() = default;
  plane_tier_definitions(
      const tf::intersect::graph::plane_tables<Index, Int> &world,
      const tf::intersect::graph::plane_tables<Index, Int> &local,
      bool local_block)
      : _immutable(&world), _local(&local), _canon_base(world.n_canon()),
        _local_block(local_block) {}

  /// The world's own canonical extent: the flat group space's frontier.
  auto canon_base() const -> Index { return _canon_base; }
  auto immutable(Index row) const -> bool {
    return !_local_block || row < Index(0);
  }
  /// The row a block of THIS arrangement names one world row by.
  static auto carried(Index world_row) -> Index { return -1 - world_row; }
  auto world_row_of(Index row) const -> Index {
    return _local_block ? -1 - row : row;
  }
  auto contains(Index row) const -> bool {
    if (immutable(row)) {
      const auto world_row = world_row_of(row);
      return world_row >= Index(0) &&
             std::size_t(world_row) < _immutable->defs().size();
    }
    return std::size_t(row) < _local->defs().size();
  }
  auto operator[](std::size_t row) const -> const def_t & {
    const auto id = Index(row);
    return immutable(id)
               ? _immutable->defs()[std::size_t(world_row_of(id))]
               : _local->defs()[row];
  }
  /// The piece ticket a row's group publishes: the immutable prefix while the
  /// world is still its authority, the PA-owned suffix once it is not.
  auto ticket(Index row) const -> Index {
    const auto group = (*this)[std::size_t(row)].id;
    return immutable(row) ? group : _canon_base + group;
  }

private:
  // the tables, not their storage: a wave grows the local buffer while the
  // reader is alive, and the world's canonical extent is frozen for the build
  const tf::intersect::graph::plane_tables<Index, Int> *_immutable = nullptr;
  const tf::intersect::graph::plane_tables<Index, Int> *_local = nullptr;
  Index _canon_base = 0;
  bool _local_block = false;
};

/// The reader one plane block resolves through: `local_block` says whether the
/// block is this arrangement's own.
template <typename Index, typename Int>
auto make_plane_tier_definitions(
    const tf::intersect::graph::plane_tables<Index, Int> &world,
    const tf::intersect::graph::plane_tables<Index, Int> &local,
    bool local_block) -> plane_tier_definitions<Index, Int> {
  return plane_tier_definitions<Index, Int>(world, local, local_block);
}

/// THE TWO TIERS' CANONICAL GROUPS IN ONE TICKET SPACE — the same space the
/// published pieces answer in. A carrier whose constraints come from both
/// tiers reads them alike, and a statement's tier is the ticket's own.
template <typename Index, typename Int> class plane_tier_groups {
public:
  plane_tier_groups() = default;
  plane_tier_groups(
      const tf::intersect::graph::plane_tables<Index, Int> &world,
      const tf::intersect::graph::plane_tables<Index, Int> &local)
      : _immutable(&world), _local(&local), _canon_base(world.n_canon()) {}

  auto canon_base() const -> Index { return _canon_base; }
  auto tier_of(Index ticket) const -> Index {
    return ticket < _canon_base ? Index(0) : Index(1);
  }
  auto group_of(Index ticket) const -> Index {
    return ticket < _canon_base ? ticket : ticket - _canon_base;
  }
  auto canon_group(Index ticket) const {
    return ticket < _canon_base
               ? _immutable->canon_group(ticket)
               : _local->canon_group(ticket - _canon_base);
  }

private:
  const tf::intersect::graph::plane_tables<Index, Int> *_immutable = nullptr;
  const tf::intersect::graph::plane_tables<Index, Int> *_local = nullptr;
  Index _canon_base = 0;
};

template <typename Index, typename Int>
auto make_plane_tier_groups(
    const tf::intersect::graph::plane_tables<Index, Int> &world,
    const tf::intersect::graph::plane_tables<Index, Int> &local)
    -> plane_tier_groups<Index, Int> {
  return plane_tier_groups<Index, Int>(world, local);
}

} // namespace tf::arrangement

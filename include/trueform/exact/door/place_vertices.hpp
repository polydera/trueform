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

#include "../../core/algorithm/block_reduce_sequenced_aggregate.hpp"
#include "../../core/algorithm/parallel_fill.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/checked.hpp"
#include "../../core/point.hpp"
#include "../../core/small_vector.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../canonical_plane.hpp"
#include "./commit_contact_planes.hpp"
#include "./owed_planes.hpp"
#include "./place_vertex.hpp"
#include "./placement_census.hpp"
#include "./placement_tables.hpp"

#include <cstddef>

namespace tf::exact::door {

/// One vertex the band moved, by flat id.
template <typename Index, typename Int> struct vertex_move {
  Index flat;
  tf::point<Int, 3> point;
};

/// THEN THE VERTICES DROP. Every vertex that owes a committed plane — of a
/// face it belongs to, or of a face it was admitted against — is placed on
/// all of them within the band, one traversal of the flat vertex space. A
/// vertex that cannot stand on all of them keeps what its own faces owe, so
/// a face that committed stays on its plane; one that cannot stand even on
/// those keeps its input point, as a `pinned` vertex always does. Both
/// refusals are counted.
///
/// The product is sparse: `moved` holds the vertices that moved, in flat
/// order, and `ticket` maps a flat id to its row or `-1`. Both stay empty
/// when nothing moved, which is the band answering exactly as zero does.
template <typename Index, typename Int, typename RealType>
auto place_vertices(const placement_tables<Index, Int, RealType> &tables,
                    const committed_planes<Index, Int> &committed,
                    Int tolerance, const tf::buffer<char> &pinned,
                    tf::buffer<Index> &ticket,
                    tf::buffer<tf::point<Int, 3>> &moved,
                    placement_census &census) -> void {
  using plane_t = tf::exact::canonical_plane<Int>;
  const auto n_vertices = tables.points.size();
  const auto no_group = [](std::size_t) { return Index(0); };

  struct local_t {
    tf::buffer<vertex_move<Index, Int>> moves;
    placement_census census{};
    tf::small_vector<plane_t, 8> owed;
    tf::small_vector<plane_t, 16> helpers;
    plane_landing<Int> landing;
  };
  struct result_t {
    tf::buffer<vertex_move<Index, Int>> moves;
    placement_census census{};
  } result;

  const auto stands_on = [](const auto &owed, const tf::point<Int, 3> &at) {
    for (const auto &plane : owed)
      if (tf::exact::orient3d_plane_sign<Int>(plane, at) != 0)
        return false;
    return true;
  };
  tf::blocked_reduce_sequenced_aggregate(
      tf::make_sequence_range(n_vertices), result, local_t{},
      [&](const auto &block, local_t &local) {
        local.moves.clear();
        local.census = placement_census{};
        for (const auto v : block) {
          for (auto k = tables.contact_offsets[v];
               k < tables.contact_offsets[v + 1]; ++k)
            if (committed.plane_of_face[std::size_t(
                    tables.contacts[std::size_t(k)])] < 0)
              ++local.census.unheld_contacts;
          const auto &original = tables.points[v];
          const auto row = committed.landing_of.size() != 0
                               ? committed.landing_of[v]
                               : Index(-1);
          tf::point<Int, 3> at = original;
          bool placed = false, whole = true;
          if (pinned.size() != 0 && pinned[v]) {
            gather_owed_planes(tables, committed.plane_of_face, committed.plane,
                               no_group, Index(-1), v, true, local.owed,
                               local.helpers);
            if (!stands_on(local.owed, original))
              ++local.census.refused;
            continue;
          }
          // every plane it owes, then only what its own faces owe
          for (const bool contacts : {true, false}) {
            gather_owed_planes(tables, committed.plane_of_face, committed.plane,
                               no_group, Index(-1), v, contacts, local.owed,
                               local.helpers);
            if (local.owed.empty()) {
              placed = true;
              break;
            }
            if (row != Index(-1) &&
                stands_on(local.owed, committed.landing[std::size_t(row)])) {
              at = committed.landing[std::size_t(row)];
              placed = true;
              break;
            }
            const auto placement = place_vertex(
                original, local.owed, local.helpers, tolerance, local.landing);
            if (placement.placed) {
              at = placement.point;
              placed = true;
              break;
            }
            whole = false;
          }
          if (!placed || !whole)
            ++local.census.refused;
          if (at != original) {
            ++local.census.moved;
            local.moves.push_back({Index(v), at});
          }
        }
      },
      [](const local_t &local, result_t &into) {
        const auto from = into.moves.size();
        into.moves.reallocate(from + local.moves.size());
        std::copy(local.moves.begin(), local.moves.end(),
                  into.moves.begin() + from);
        into.census.moved += local.census.moved;
        into.census.refused += local.census.refused;
        into.census.unheld_contacts += local.census.unheld_contacts;
      });
  census.moved = result.census.moved;
  census.refused = result.census.refused;
  census.unheld_contacts = result.census.unheld_contacts;

  const auto &moves = result.moves;
  ticket.clear();
  moved.allocate(moves.size());
  if (moves.size() == 0)
    return;
  ticket.allocate(n_vertices);
  tf::parallel_fill(ticket, Index(-1));
  tf::parallel_for_each(
      tf::make_sequence_range(moves.size()),
      [&ticket, &moved, &moves](std::size_t r) {
        ticket[std::size_t(moves[r].flat)] = Index(r);
        moved[r] = moves[r].point;
      },
      tf::checked);
}

} // namespace tf::exact::door

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
#include "../admitted_tet_edge.hpp"
#include "../../../../core/config/verify.hpp"
#include "../find_tet_direction.hpp"
#include "../find_tet_edge.hpp"
#include "../flip_tet_ring.hpp"
#include "../tet_facet_is_protected.hpp"
#include "../tet_facet_key.hpp"
#include "../tet_shared_owners.hpp"
#include "../remove_tet_edge_by_flips.hpp"
#include "../remove_tet_facet_by_flips.hpp"
#include "../tet_deepest_of.hpp"
#include "../tet_edge_ref.hpp"
#include "../tet_flip_constraints.hpp"
#include "../tet_flip_scratch.hpp"
#include "../tet_state.hpp"
#include "../tet_triangle_edge_contact.hpp"
#include "../tet_with_sites.hpp"
#include <array>
#include <cstddef>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// Recover the edge from `start` to `end` by flips: from `start` the first
/// facet or edge the segment crosses is removed (@ref
/// remove_tet_facet_by_flips, @ref remove_tet_edge_by_flips) under the segment
/// as constraint, and the path is read again, until the edge stands; under
/// `fullsearch` every facet and edge along the segment is tried in turn before
/// giving up. A STOPPED TASK STANDS WHERE IT IS: the moment the admission
/// stops it (@ref admit_tet_row) the walk begins no further removal and
/// answers that the edge does not stand, so the round defers it whole and
/// the frontier walks it again. THE TURNS ARE THE WALK'S BOUND: a turn removes one obstruction
/// between the ends and reads the path again, so a walk takes as many turns
/// as its segment crosses cells, and a walk that spends
/// `scratch.walk_passes` of them counts itself in `scratch.walk_capped` and
/// gives up, which refuses the row rather than escalating it. A search reads
/// the cell it walks into, or spins the ring it
/// walks into ONCE, for the contact the segment makes with it; a complex
/// whose link states one gives the contact, and a search that finds none ends
/// the walk as a lost direction does. Returns whether the edge stands; `on` names a site the segment
/// passes through, which the edge cannot be recovered past, `-1` otherwise,
/// and `crossed` the first segment edge the walk met crossing it, `-1`
/// twice otherwise; `pierced` the face owning the first protected facet the
/// walk met the segment piercing, `-1` where it met none.
template <typename Owner, typename IsSegment>
auto recover_tet_edge_by_flips(
    Owner &owner, tet_flip_scratch<typename Owner::index_type> &scratch,
    typename Owner::index_type start, typename Owner::index_type end,
    bool fullsearch, const IsSegment &is_segment,
    typename Owner::index_type &on,
    std::array<typename Owner::index_type, 2> &crossed,
    typename Owner::index_type &pierced) -> bool {
  using Index = typename Owner::index_type;
  using Int = typename Owner::int_type;
  using ref_t = tet_edge_ref<Index>;
  pierced = Index(-1);
  const auto pierce = [&owner, &pierced](const ref_t &facet) {
    const std::size_t slot = tet_edge_oppo(facet);
    if (pierced != Index(-1) ||
        !tet_facet_is_protected(owner, facet.cell, slot))
      return;
    const Index who = owner._owners[4 * std::size_t(facet.cell) + slot];
    if (who == Owner::shared_owner) {
      const auto stack = tet_shared_owners(
          owner, tet_facet_key(owner._corners[std::size_t(facet.cell)], slot));
      if (stack.first != stack.second)
        pierced = stack.first->face;
    } else if (who != Owner::none)
      pierced = who;
  };
  tet_flip_constraints<Index, typename Owner::coord_type, Int> fc;
  fc.segment[0] = start;
  fc.segment[1] = end;
  on = Index(-1);
  crossed = {Index(-1), Index(-1)};
  const auto corners = [&owner](const ref_t &r) {
    return tet_edge_corners(owner, r);
  };
  const auto meet = [&corners, &is_segment, &crossed](const ref_t &across) {
    const auto c = corners(across);
    if (crossed[0] == Index(-1) && is_segment(c[0], c[1]))
      crossed = {c[0], c[1]};
  };
  const auto met = [&](const ref_t &facet, tet_triangle_edge_contact &out) {
    const auto c = corners(facet);
    if (c[0] == Owner::infinite || c[1] == Owner::infinite ||
        c[2] == Owner::infinite)
      return 0;
    return tet_with_sites(
        owner, tet_deepest_of(owner, c[0], c[1], c[2], start, end),
        [&](const auto &sites) {
          using lattice = typename std::decay_t<decltype(sites)>::int_type;
          return classify_tet_triangle_edge<lattice>(
              sites[std::size_t(c[0])].pt, sites[std::size_t(c[1])].pt,
              sites[std::size_t(c[2])].pt, sites[std::size_t(start)].pt,
              sites[std::size_t(end)].pt, out);
        });
  };
  std::size_t turns = 0;
  const auto spent = [&scratch, &turns] {
    if (turns++ < scratch.walk_passes)
      return false;
    ++scratch.walk_capped;
    return true;
  };
  for (;;) {
    if (spent() || scratch.round.stopped)
      return false;
    auto edge = admitted_tet_site_handle(owner, scratch.round, start);
    if (scratch.round.stopped)
      return false;
    auto dir = find_tet_direction(owner, edge, end);
    if (dir == tet_direction::across_vertex) {
      const Index dest = corners(edge)[1];
      if (dest == end)
        return true;
      on = dest;
      return false;
    }
    if (dir == tet_direction::hull || dir == tet_direction::lost)
      return false;
    edge = tet_edge_turn_sym(edge);
    if (dir == tet_direction::across_face) {
      pierce(edge);
      if (remove_tet_facet_by_flips(owner, scratch, edge, fc, is_segment))
        continue;
    } else {
      meet(edge);
      if (remove_tet_edge_by_flips(owner, scratch, edge, fc, is_segment) == 2)
        continue;
    }
    if (!fullsearch)
      return false;
    edge = admitted_tet_site_handle(owner, scratch.round, start);
    if (scratch.round.stopped)
      return false;
    dir = find_tet_direction(owner, edge, end);
    if constexpr (tf::core::verification)
      tf::core::verify(dir == tet_direction::across_face ||
                           dir == tet_direction::across_edge,
                       "an edge that does not stand leaves its start across a "
                       "face or an edge");
    edge = tet_edge_turn_sym(edge);
    bool success = false;
    for (;;) {
      if (spent() || scratch.round.stopped)
        return false;
      edge = admitted_tet_edge_across(owner, scratch.round, edge);
      tet_triangle_edge_contact contact{};
      tet_contact kind = tet_contact::disjoint;
      int pos = 0;
      ref_t neigh = edge;
      if (dir == tet_direction::across_face) {
        const auto cell_corners = owner._corners[std::size_t(edge.cell)];
        const std::size_t entered = tet_edge_oppo(edge);
        for (std::size_t s = 0; s < 4 && kind == tet_contact::disjoint; ++s) {
          if (s == entered)
            continue;
          const auto f = tet_facet(cell_corners, s);
          neigh = tet_face_handle(owner, edge.cell, f[0], f[1], f[2]);
          if (met(neigh, contact)) {
            kind = contact.type[0];
            pos = contact.pos[0];
          }
        }
      } else {
        const Index ring = edge.cell;
        for (;;) {
          for (std::size_t i = 0; i < 2 && kind == tet_contact::disjoint; ++i) {
            neigh = i == 0 ? tet_edge_turn_sym(edge) : tet_edge_turn_back_sym(edge);
            if (met(neigh, contact)) {
              kind = contact.type[0];
              pos = contact.pos[0];
            }
          }
          if (kind != tet_contact::disjoint)
            break;
          edge = admitted_tet_edge_next(owner, scratch.round, edge);
          if (edge.cell == ring || scratch.round.stopped)
            break;
        }
      }
      if constexpr (tf::core::verification)
        tf::core::verify(kind != tet_contact::disjoint,
                         "the cell or the ring the segment walks into meets it");
      if (kind == tet_contact::disjoint)
        break;
      for (int i = 0; i < pos; ++i)
        neigh = tet_edge_turn(neigh);
      if (kind == tet_contact::share_vertex) {
        if (corners(neigh)[0] != end)
          on = corners(neigh)[0];
        break;
      }
      if (kind == tet_contact::across_vertex) {
        on = corners(neigh)[0];
        break;
      }
      edge = neigh;
      const auto bak = corners(edge);
      const Index bak_cell = edge.cell;
      bool flipped = false;
      if (kind == tet_contact::across_face) {
        dir = tet_direction::across_face;
        pierce(edge);
        flipped = remove_tet_facet_by_flips(owner, scratch, edge, fc, is_segment);
      } else {
        if constexpr (tf::core::verification)
          tf::core::verify(
              kind == tet_contact::across_edge,
              "a contact neither a facet nor a vertex crosses an edge");
        dir = tet_direction::across_edge;
        meet(edge);
        flipped =
            remove_tet_edge_by_flips(owner, scratch, edge, fc, is_segment) == 2;
      }
      if (flipped) {
        success = true;
        break;
      }
      if (owner._states[std::size_t(bak_cell)] != tet_state::live ||
          corners(edge) != bak) {
        auto seat = admitted_tet_site_handle(owner, scratch.round, bak[0]);
        if (scratch.round.stopped)
          break;
        const auto again = find_tet_direction(owner, seat, bak[1]);
        bool found = false;
        if (again == tet_direction::across_vertex && corners(seat)[1] == bak[1]) {
          auto spin = seat;
          do {
            if (corners(spin)[2] == bak[2]) {
              found = true;
              break;
            }
            spin = admitted_tet_edge_next(owner, scratch.round, spin);
          } while (spin.cell != seat.cell && !scratch.round.stopped);
          if (found && corners(spin)[3] != bak[3]) {
            spin = tet_edge_across(owner, spin);
            found = corners(spin)[3] == bak[3];
          }
          if (found)
            edge = spin;
        }
        if (!found)
          break;
      }
    }
    if (!success)
      return false;
  }
}

} // namespace tf::topology::cdt::dt3

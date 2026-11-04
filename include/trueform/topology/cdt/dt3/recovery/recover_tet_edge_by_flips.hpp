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
#include "../find_tet_direction.hpp"
#include "../find_tet_edge.hpp"
#include "../flip_tet_ring.hpp"
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
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace tf::topology::cdt::dt3 {

/// Recover the edge from `start` to `end` by flips: from `start` the first
/// facet or edge the segment crosses is removed (@ref
/// remove_tet_facet_by_flips, @ref remove_tet_edge_by_flips) under the segment
/// as constraint, and the path is read again, until the edge stands; under
/// `fullsearch` every facet and edge along the segment is tried in turn before
/// giving up. Returns whether the edge stands; `on` names a site the segment
/// passes through, which the edge cannot be recovered past, `-1` otherwise.
template <typename Owner, typename IsSegment>
auto recover_tet_edge_by_flips(
    Owner &owner, tet_flip_scratch<typename Owner::index_type> &scratch,
    typename Owner::index_type start, typename Owner::index_type end,
    bool fullsearch, const IsSegment &is_segment,
    typename Owner::index_type &on) -> bool {
  using Index = typename Owner::index_type;
  using Int = typename Owner::int_type;
  using ref_t = tet_edge_ref<Index>;
  tet_flip_constraints<Index, typename Owner::coord_type, Int> fc;
  fc.segment[0] = start;
  fc.segment[1] = end;
  on = Index(-1);
  const auto corners = [&owner](const ref_t &r) {
    return tet_edge_corners(owner, r);
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
  for (;;) {
    auto edge = tet_site_handle(owner, start);
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
      if (remove_tet_facet_by_flips(owner, scratch, edge, fc, is_segment))
        continue;
    } else if (remove_tet_edge_by_flips(owner, scratch, edge, fc, is_segment) ==
               2)
      continue;
    if (!fullsearch)
      return false;
    edge = tet_site_handle(owner, start);
    dir = find_tet_direction(owner, edge, end);
    assert(dir == tet_direction::across_face ||
           dir == tet_direction::across_edge);
    edge = tet_edge_turn_sym(edge);
    bool success = false;
    for (;;) {
      edge = tet_edge_across(owner, edge);
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
        assert(kind != tet_contact::disjoint);
      } else {
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
          edge = tet_edge_next(owner, edge);
        }
      }
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
        flipped = remove_tet_facet_by_flips(owner, scratch, edge, fc, is_segment);
      } else {
        assert(kind == tet_contact::across_edge);
        dir = tet_direction::across_edge;
        flipped =
            remove_tet_edge_by_flips(owner, scratch, edge, fc, is_segment) == 2;
      }
      if (flipped) {
        success = true;
        break;
      }
      if (owner._states[std::size_t(bak_cell)] != tet_state::live ||
          corners(edge) != bak) {
        auto seat = tet_site_handle(owner, bak[0]);
        const auto again = find_tet_direction(owner, seat, bak[1]);
        bool found = false;
        if (again == tet_direction::across_vertex && corners(seat)[1] == bak[1]) {
          auto spin = seat;
          do {
            if (corners(spin)[2] == bak[2]) {
              found = true;
              break;
            }
            spin = tet_edge_next(owner, spin);
          } while (spin.cell != seat.cell);
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

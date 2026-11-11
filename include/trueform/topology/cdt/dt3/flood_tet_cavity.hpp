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
#include "./close_tet_cavity_links.hpp"
#include "./open_tet_cavity_links.hpp"
#include "./record_tet_killed_edges.hpp"
#include "./state_tet_cavity_facet.hpp"
#include "./tet_cavity_sides.hpp"
#include "./tet_conflicts.hpp"
#include "./tet_facet_is_protected.hpp"
#include "./tet_facet_sees.hpp"
#include "./tet_facets.hpp"
#include "./tet_protected_count.hpp"
#include "./tet_site_position.hpp"
#include "./tet_state.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

/// Mark the conflict cavity dead and record its boundary from surviving
/// cells. A dead neighbor is already visited by this flood; a facet
/// `barred(cell, slot)` names is a wall whatever its far cell's verdict. A
/// cavity holding both cells of a barred facet wraps it, and so does one
/// bounded by a barred facet the site does not strictly see: the flood went
/// around the wall, and the star would invert. On a constrained Delaunay
/// complex an unbarred boundary facet is locally Delaunay, so the site sees
/// it; a protected lane may hold a complex that is not, around a constraint
/// whose recovery was refused, so under a lane, `Constrained`, every finite
/// boundary facet the site does not strictly see wraps the same way and the
/// star never inverts. A wrapped cavity is given back live and the flood
/// returns false. A finite boundary facet no lane bars that the site does
/// not strictly see is the non-Delaunay complex's own: the cell across it
/// leaves the cavity, the boundary is restated from the cells left, and the
/// reading repeats until every facet is seen; the cavity that remains holds
/// the site's cell and is star-shaped from it, an empty one wraps.
template <bool Constrained, typename Owner, typename Barred>
auto flood_tet_cavity(Owner &owner, typename Owner::index_type seed,
                      typename Owner::index_type site, const Barred &barred)
    -> bool {
  using Index = typename Owner::index_type;
  auto &cavity = owner._cavity;
  auto &entered = owner._entered;
  auto &boundary = owner._boundary;
  cavity.clear();
  entered.clear();
  boundary.clear();
  open_tet_cavity_links(owner._links, tet_site_count(owner));
  owner._states[std::size_t(seed)] = tet_state::dead;
  cavity.push_back(seed);
  entered.push_back(std::uint8_t(tet_cavity_seed_entrance));
  bool wrapped = false;
  for (std::size_t head = 0; head < cavity.size(); ++head) {
    const Index cell = cavity[head];
    const std::size_t entrance = entered[head];
    const auto *asked = tet_cavity_sides[entrance];
    const std::size_t n_asked =
        3 + std::size_t(entrance == tet_cavity_seed_entrance);
    if constexpr (!Constrained) {
      // Both outcomes of a verdict are an append: each is written at its
      // lane's end and kept by the step of that end, so no branch reads the
      // verdict. A side is asked again for its state, as a cell that two
      // slots name is dead by the second.
      int verdict[4];
      for (std::size_t k = 0; k < n_asked; ++k) {
        const Index side = owner._neighbors[std::size_t(cell)][asked[k]];
        verdict[k] =
            owner._states[std::size_t(side)] == tet_state::dead
                ? 0
                : tet_conflicts(owner, side, site, [](Index) { return true; });
      }
      std::size_t n_cavity = cavity.size(), n_boundary = boundary.size();
      cavity.reallocate(n_cavity + n_asked);
      entered.reallocate(n_cavity + n_asked);
      boundary.reallocate(n_boundary + n_asked);
      for (std::size_t k = 0; k < n_asked; ++k) {
        const Index side = owner._neighbors[std::size_t(cell)][asked[k]];
        const tet_state state = owner._states[std::size_t(side)];
        const bool live = state != tet_state::dead;
        const bool inside = live & (verdict[k] > 0);
        const auto sides = owner._neighbors[std::size_t(side)];
        const std::size_t mate = std::size_t(sides[1] == cell) +
                                 2 * std::size_t(sides[2] == cell) +
                                 3 * std::size_t(sides[3] == cell);
        cavity[n_cavity] = side;
        entered[n_cavity] = std::uint8_t(mate);
        n_cavity += std::size_t(inside);
        owner._states[std::size_t(side)] = inside ? tet_state::dead : state;
        boundary[n_boundary] = Index(4 * std::size_t(side) + mate);
        n_boundary += std::size_t(live & !inside);
      }
      cavity.reallocate(n_cavity);
      entered.reallocate(n_cavity);
      boundary.reallocate(n_boundary);
    } else {
      for (std::size_t k = 0; k < n_asked; ++k) {
        const std::size_t slot = asked[k];
        const Index side = owner._neighbors[std::size_t(cell)][slot];
        const bool wall = barred(cell, slot);
        if (owner._states[std::size_t(side)] == tet_state::dead) {
          wrapped = wrapped || wall;
          continue;
        }
        const auto sides = owner._neighbors[std::size_t(side)];
        const std::size_t mate = std::size_t(sides[1] == cell) +
                                 2 * std::size_t(sides[2] == cell) +
                                 3 * std::size_t(sides[3] == cell);
        if (!wall &&
            tet_conflicts(owner, side, site, [](Index) { return true; })) {
          owner._states[std::size_t(side)] = tet_state::dead;
          cavity.push_back(side);
          entered.push_back(std::uint8_t(mate));
          continue;
        }
        boundary.push_back(Index(4 * std::size_t(side) + mate));
        state_tet_cavity_facet(owner._links,
                               tet_facet(owner._corners[side], mate));
      }
    }
  }
  if constexpr (!Constrained)
    for (auto ticket : boundary)
      state_tet_cavity_facet(owner._links,
                             tet_facet(owner._corners[std::size_t(ticket) / 4],
                                       std::size_t(ticket) % 4));
  if constexpr (Constrained)
    for (;;) {
      bool shrunk = false;
      for (auto ticket : boundary) {
        const Index cell = Index(std::size_t(ticket) / 4);
        const std::size_t slot = std::size_t(ticket) % 4;
        const bool open =
            owner._corners[std::size_t(cell)][3] == Owner::infinite &&
            slot != 3;
        if (owner._states[std::size_t(cell)] == tet_state::dead) {
          wrapped = true;
          continue;
        }
        if (open || tet_facet_sees(owner, cell, slot, site))
          continue;
        if (barred(cell, slot)) {
          wrapped = true;
          continue;
        }
        const Index across = owner._neighbors[std::size_t(cell)][slot];
        owner._states[std::size_t(across)] =
            owner._corners[std::size_t(across)][3] == Owner::infinite
                ? tet_state::infinite
                : tet_state::live;
        shrunk = true;
      }
      if (wrapped || !shrunk)
        break;
      std::size_t kept = 0;
      for (auto cell : cavity)
        if (owner._states[std::size_t(cell)] == tet_state::dead)
          cavity[kept++] = cell;
      cavity.erase_till_end(cavity.begin() + std::ptrdiff_t(kept));
      if (!cavity.size()) {
        wrapped = true;
        break;
      }
      boundary.clear();
      open_tet_cavity_links(owner._links, tet_site_count(owner));
      for (auto cell : cavity)
        for (std::size_t slot = 0; slot < 4; ++slot) {
          const Index side = owner._neighbors[std::size_t(cell)][slot];
          if (owner._states[std::size_t(side)] == tet_state::dead)
            continue;
          const auto sides = owner._neighbors[std::size_t(side)];
          const std::size_t mate = std::size_t(sides[1] == cell) +
                                   2 * std::size_t(sides[2] == cell) +
                                   3 * std::size_t(sides[3] == cell);
          boundary.push_back(Index(4 * std::size_t(side) + mate));
          state_tet_cavity_facet(owner._links,
                                 tet_facet(owner._corners[side], mate));
        }
    }
  if (wrapped) {
    for (auto cell : cavity)
      owner._states[std::size_t(cell)] =
          owner._corners[std::size_t(cell)][3] == Owner::infinite
              ? tet_state::infinite
              : tet_state::live;
    return false;
  }
  close_tet_cavity_links(owner._links);
  owner._n_dead += cavity.size();
  if (owner._record_killed)
    record_tet_killed_edges(owner, cavity, owner._killed);
  return true;
}

/// The flood the protected lane bars. An empty lane over a Delaunay complex
/// floods unbarred and never wraps; one a flip has left non-Delaunay
/// (`_delaunay`) rechecks what the site sees, under no walls.
template <typename Owner>
auto flood_tet_cavity(Owner &owner, typename Owner::index_type seed,
                      typename Owner::index_type site) -> bool {
  using Index = typename Owner::index_type;
  if (tet_protected_count(owner) == 0) {
    const auto open = [](Index, std::size_t) { return false; };
    if (owner._delaunay)
      return flood_tet_cavity<false>(owner, seed, site, open);
    return flood_tet_cavity<true>(owner, seed, site, open);
  }
  return flood_tet_cavity<true>(
      owner, seed, site, [&owner](Index cell, std::size_t slot) {
        return tet_facet_is_protected(owner, cell, slot);
      });
}

} // namespace tf::topology::cdt::dt3

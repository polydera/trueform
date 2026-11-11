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
#include "../../../core/config/verify.hpp"
#include "../../../core/buffer.hpp"
#include "./record_tet_written_row.hpp"
#include "./state_tet_facet_protection.hpp"
#include "./tet_facet_is_protected.hpp"
#include "./tet_facets.hpp"
#include "./tet_orient3d_sites.hpp"
#include "./tet_site_position.hpp"
#include "./tet_state.hpp"
#include "./walk_straight_tet.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Retire the site `site` into the site `survivor`, the collapse of the
/// edge between them, the inverse of a split: every cell holding both
/// vanishes, its two facets holding one of them closing onto one, and every
/// other cell holding the site is rewritten in place onto the survivor, its
/// slots, links and protections standing. The survivor keeps its position.
/// The collapse is exact and all or nothing: it refuses, changing nothing,
/// where the survivor is not on an edge of the site's star, where a
/// rewritten cell would not stand positive on the lattice, where two facets
/// closing onto one are held by different faces or one is a stacked facet,
/// where a held facet of the star is a stack, and where the site's star
/// reaches the hull — the one refusal no statement of the census resolves.
/// A held facet holding both sites vanishes with its cells and its face, as
/// every face whose held facet the collapse moved, is named in
/// `_collapse.faces`, ascending, for the face phase to read again. The
/// rewritten rows are recorded and the site's edges are killed; its slot,
/// floor and deep row stay, named by no cell. The retirement is the
/// caller's to state, the site into the survivor, with its identity
/// authority, and the inputs naming it are renamed once, after the
/// statement's collapses, by @ref resolve_tet_retired_sites. The rewritten
/// star is not restored to Delaunay: that is the restoration's work, as
/// after any insertion. `consents(star)`, asked with the site's star once
/// every check has passed and before anything is written, is the caller's
/// own refusal. Each pass costs the site's star.
template <typename Owner, typename Consents>
auto collapse_tet_site(Owner &owner, typename Owner::index_type site,
                       typename Owner::index_type survivor,
                       const Consents &consents) -> bool {
  using Index = typename Owner::index_type;
  auto &work = owner._collapse;
  auto &star = work.star;
  auto &ring = work.ring;
  auto &faces = work.faces;
  star.clear();
  ring.clear();
  faces.clear();
  if (site == survivor)
    return false;
  const auto slot_of = [&owner](Index cell, Index corner) {
    std::size_t k = 0;
    while (k < 4 && owner._corners[std::size_t(cell)][k] != corner)
      ++k;
    if constexpr (tf::core::verification)
      tf::core::verify(k < 4, "the cell holds the corner");
    return k;
  };
  const auto back_slot = [&owner](Index from, Index to) {
    std::size_t k = 0;
    while (k < 4 && owner._neighbors[std::size_t(from)][k] != to)
      ++k;
    if constexpr (tf::core::verification)
      tf::core::verify(k < 4, "the mate names the cell back");
    return k;
  };
  const auto holds = [&owner](Index cell, Index corner) {
    for (const auto c : owner._corners[std::size_t(cell)])
      if (c == corner)
        return true;
    return false;
  };
  const auto held = [&owner](Index cell, std::size_t slot) {
    return std::size_t(cell) < owner._protected_slots.size() &&
           tet_facet_is_protected(owner, cell, slot);
  };
  const auto face_at = [&owner](Index cell, std::size_t slot) {
    return owner._owners[4 * std::size_t(cell) + slot];
  };

  // the star, flooded across the facets holding the site
  if (work.marks.size() < std::size_t(owner._corners.size())) {
    const std::size_t known = work.marks.size();
    work.marks.reallocate(std::size_t(owner._corners.size()));
    std::fill(work.marks.begin() + std::ptrdiff_t(known), work.marks.end(),
              std::uint32_t(0));
  }
  if (++work.generation == 0) {
    std::fill(work.marks.begin(), work.marks.end(), std::uint32_t(0));
    work.generation = 1;
  }
  const auto admit = [&work, &star](Index cell) {
    auto &mark = work.marks[std::size_t(cell)];
    if (mark == work.generation)
      return;
    mark = work.generation;
    star.push_back(cell);
  };
  const Index start = walk_straight_tet(owner, site, owner._hint);
  if (!holds(start, site))
    return false;
  admit(start);
  for (std::size_t head = 0; head < star.size(); ++head) {
    const Index cell = star[head];
    if (owner._corners[std::size_t(cell)][3] == Owner::infinite)
      return false;
    const std::size_t at = slot_of(cell, site);
    for (std::size_t k = 0; k < 4; ++k)
      if (k != at)
        admit(owner._neighbors[std::size_t(cell)][k]);
  }

  // every check before anything is written
  for (const auto cell : star) {
    if (holds(cell, survivor)) {
      ring.push_back(cell);
      continue;
    }
    auto corners = owner._corners[std::size_t(cell)];
    std::array<Index, 4> moved{corners[0], corners[1], corners[2], corners[3]};
    moved[slot_of(cell, site)] = survivor;
    if (tet_orient3d_sites(owner, moved[0], moved[1], moved[2], moved[3]) <= 0)
      return false;
    for (std::size_t k = 0; k < 4; ++k)
      if (corners[k] != site && held(cell, k) &&
          face_at(cell, k) == Owner::shared_owner)
        return false;
  }
  if (!ring.size())
    return false;
  for (const auto cell : ring) {
    const std::size_t is = slot_of(cell, site), iu = slot_of(cell, survivor);
    const bool hx = held(cell, is), hy = held(cell, iu);
    if ((hx && face_at(cell, is) == Owner::shared_owner) ||
        (hy && face_at(cell, iu) == Owner::shared_owner) ||
        (hx && hy && face_at(cell, is) != face_at(cell, iu)))
      return false;
    for (std::size_t k = 0; k < 4; ++k)
      if (k != is && k != iu && held(cell, k) &&
          face_at(cell, k) == Owner::shared_owner)
        return false;
  }

  if (!consents(static_cast<const tf::buffer<Index> &>(star)))
    return false;

  // the faces whose held facets move or vanish
  for (const auto cell : star)
    for (std::size_t k = 0; k < 4; ++k)
      if (owner._corners[std::size_t(cell)][k] != site && held(cell, k) &&
          face_at(cell, k) != Owner::none)
        faces.push_back(face_at(cell, k));
  for (const auto cell : ring) {
    const std::size_t is = slot_of(cell, site);
    if (held(cell, is) && face_at(cell, is) != Owner::none)
      faces.push_back(face_at(cell, is));
  }
  std::sort(faces.begin(), faces.end());
  faces.erase_till_end(std::unique(faces.begin(), faces.end()));

  // the site's edges, which the collapse takes
  if (owner._record_killed) {
    auto &edges = work.edges;
    edges.clear();
    for (const auto cell : star)
      for (const auto corner : owner._corners[std::size_t(cell)])
        if (corner != site)
          edges.push_back({std::min(site, corner), std::max(site, corner)});
    std::sort(edges.begin(), edges.end());
    edges.erase_till_end(std::unique(edges.begin(), edges.end()));
    for (const auto &edge : edges)
      owner._killed.push_back(edge);
  }

  // the ring vanishes: its held facets released, its two facets holding one
  // of the sites closed onto one, carrying the protection either held
  for (const auto cell : ring) {
    const std::size_t is = slot_of(cell, site), iu = slot_of(cell, survivor);
    const Index face =
        held(cell, is) ? face_at(cell, is)
                       : (held(cell, iu) ? face_at(cell, iu) : Owner::none);
    const bool keep = held(cell, is) || held(cell, iu);
    for (std::size_t k = 0; k < 4; ++k)
      if (held(cell, k))
        state_tet_facet_protection(owner, cell, k, false, Owner::none);
    const Index x = owner._neighbors[std::size_t(cell)][is];
    const Index y = owner._neighbors[std::size_t(cell)][iu];
    const std::size_t bx = back_slot(x, cell), by = back_slot(y, cell);
    owner._neighbors[std::size_t(x)][bx] = y;
    owner._neighbors[std::size_t(y)][by] = x;
    if (keep)
      state_tet_facet_protection(owner, x, bx, true, face);
    owner._states[std::size_t(cell)] = tet_state::dead;
    ++owner._n_dead;
  }

  // the rest of the star, rewritten in place onto the survivor
  Index kept = Owner::none;
  for (const auto cell : star) {
    if (owner._states[std::size_t(cell)] == tet_state::dead)
      continue;
    owner._corners[std::size_t(cell)][slot_of(cell, site)] = survivor;
    if (owner._record_written)
      record_tet_written_row(owner, cell);
    if (owner._record_site_cells)
      for (const auto corner : owner._corners[std::size_t(cell)])
        owner._site_cells.store(std::size_t(corner), cell);
    kept = cell;
  }
  if (owner._record_site_cells)
    owner._site_cells.store(std::size_t(site), Owner::none);
  if (kept != Owner::none)
    owner._hint = kept;

  owner._delaunay = false;
  owner._canonical = false;
  return true;
}

template <typename Owner>
auto collapse_tet_site(Owner &owner, typename Owner::index_type site,
                       typename Owner::index_type survivor) -> bool {
  return collapse_tet_site(owner, site, survivor,
                           [](const auto &) { return true; });
}

} // namespace tf::topology::cdt::dt3

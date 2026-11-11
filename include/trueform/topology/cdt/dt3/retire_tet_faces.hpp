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
#include "../../../core/config/verify.hpp"
#include "../../../core/none.hpp"
#include "./recovery/enter_tet_site_star.hpp"
#include "./recovery/gather_tet_star.hpp"
#include "./recovery/open_tet_stamps.hpp"
#include "./recovery/open_tet_star_stamps.hpp"
#include "./state_tet_facet_protection.hpp"
#include "./tet_facet_is_protected.hpp"
#include "./tet_facet_key.hpp"
#include "./tet_facet_owner.hpp"
#include "./tet_owner_complex.hpp"
#include "./tet_shared_owners.hpp"
#include "./tet_site_position.hpp"
#include "./tet_state.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// One protected facet a retirement may change: the ticket `4 cell + slot`
/// it is read from and the owner it held.
template <typename Index> struct tet_retired_facet {
  Index ticket;
  Index was;
};

/// Retire the faces `retired`, ascending: every protected facet a retired
/// face owns loses that owner — a facet it owns alone ceases to be
/// protected on both its cells, and a stack drops the face's row, holding
/// what is left as one owner where one row of one layer is, as a stack where
/// more is, or ceasing to be protected where none is; a protected facet the
/// complex does not hold drops it alike. The cells stay as they are.
///
/// The facets are reached from the faces: `sites_of(face, emit)` states the
/// sites of each retired face's rim, and from every site reached the cells
/// about it are gathered (@ref gather_tet_star), each protected facet there
/// that a retired face owns taken once, by the lower row holding it, and its
/// corners reached in turn. A face's held facets tile its cover, which grows
/// from its rim, so every one is reached. A site stands where the cell at
/// its position holds it: a site a collapse retired holds no cell and no
/// facet, and is passed. O(the stars of the sites reached +
/// k log k) for the k facets taken — the retired faces' facets and their
/// corners' stars, never the complex. `facets` is the caller's scratch; the
/// walk is the owner's (`_retire`).
template <typename Owner, typename SitesOf>
auto retire_tet_faces(
    Owner &owner, const tf::buffer<typename Owner::index_type> &retired,
    const SitesOf &sites_of,
    tf::buffer<tet_retired_facet<typename Owner::index_type>> &facets) -> void {
  using Index = typename Owner::index_type;
  const auto is_retired = [&retired](Index face) {
    return std::binary_search(retired.begin(), retired.end(), face);
  };
  const auto owned = [&owner, &is_retired](std::size_t cell,
                                           std::size_t slot) -> bool {
    if (!tet_facet_is_protected(owner, Index(cell), slot))
      return false;
    const Index who = owner._owners[4 * cell + slot];
    if (who != Owner::shared_owner)
      return who >= 0 && is_retired(who);
    const auto stack =
        tet_shared_owners(owner, tet_facet_key(owner._corners[cell], slot));
    for (auto at = stack.first; at != stack.second; ++at)
      if (is_retired(at->face))
        return true;
    return false;
  };
  const tf::none_t serial{};
  const tet_owner_complex<false, Owner, const tf::none_t> complex(owner,
                                                                  serial);
  auto &walk = owner._retire;
  open_tet_stamps(walk.sites, tet_site_count(owner));
  open_tet_star_stamps(complex, walk.stars);
  auto &queue = walk.queue;
  queue.clear();
  const auto reach = [&walk, &queue](Index site) {
    if (take_tet_stamp(walk.sites, std::size_t(site)))
      queue.push_back(site);
  };
  for (const Index face : retired)
    sites_of(face, reach);
  facets.clear();
  const Index entry = complex.entry();
  auto &star = walk.star;
  for (std::size_t head = 0; head < queue.size(); ++head) {
    const Index site = queue[head];
    star.clear();
    bool ticketed = false;
    const Index from = enter_tet_site_star(complex, site, entry, ticketed);
    bool stands = false;
    for (const auto corner : owner._corners[std::size_t(from)])
      stands = stands || corner == site;
    if (!stands)
      continue;
    gather_tet_star(complex, site, from, walk.stars, star);
    for (const Index row : star) {
      const auto cell = std::size_t(row);
      const auto corners = owner._corners[cell];
      for (std::size_t slot = 0; slot < 4; ++slot) {
        if (corners[slot] == site || !owned(cell, slot))
          continue;
        for (std::size_t k = 0; k < 4; ++k)
          if (k != slot)
            reach(corners[k]);
        std::size_t lower = cell, at = slot;
        const auto beyond = std::size_t(owner._neighbors[cell][slot]);
        if (beyond < cell) {
          lower = beyond;
          at = 0;
          while (at < 4 && std::size_t(owner._neighbors[lower][at]) != cell)
            ++at;
          if constexpr (tf::core::verification)
            tf::core::verify(at < 4, "the mate names the cell back");
        }
        facets.push_back(
            {Index(4 * lower + at), owner._owners[4 * lower + at]});
      }
    }
  }
  std::sort(facets.begin(), facets.end(),
            [](const auto &x, const auto &y) { return x.ticket < y.ticket; });
  facets.erase_till_end(std::unique(
      facets.begin(), facets.end(),
      [](const auto &x, const auto &y) { return x.ticket == y.ticket; }));
  auto &shared = owner._shared;
  shared.erase_till_end(std::remove_if(
      shared.begin(), shared.end(),
      [&is_retired](const auto &row) { return is_retired(row.face); }));
  const auto single = [&owner](const std::array<Index, 3> &key) {
    const auto stack = tet_shared_owners(owner, key);
    return stack.second - stack.first == 1 && stack.first->layers == 1
               ? stack.first->face
           : stack.first == stack.second ? Index(-1)
                                         : Owner::shared_owner;
  };
  bool lone = false;
  for (const auto &facet : facets) {
    const Index cell = Index(std::size_t(facet.ticket) / 4);
    const std::size_t slot = std::size_t(facet.ticket) % 4;
    Index left = Owner::none;
    if (facet.was == Owner::shared_owner) {
      left = single(tet_facet_key(owner._corners[std::size_t(cell)], slot));
      if (left == Owner::shared_owner)
        continue;
      lone = lone || left >= 0;
    }
    state_tet_facet_protection(owner, cell, slot, left >= 0, left);
  }
  auto &unheld = owner._protected_unheld;
  std::size_t kept = 0;
  for (std::size_t i = 0; i < unheld.size(); ++i) {
    auto row = unheld[i];
    if (row.face == Owner::shared_owner) {
      const Index left = single(row.key);
      if (left == Owner::shared_owner) {
        unheld[kept++] = row;
        continue;
      }
      lone = lone || left >= 0;
      if (left < 0)
        continue;
      row.face = left;
    } else if (row.face >= 0 && is_retired(row.face)) {
      continue;
    }
    unheld[kept++] = row;
  }
  unheld.erase_till_end(unheld.begin() + std::ptrdiff_t(kept));
  if (lone) {
    std::size_t at = 0;
    for (std::size_t r = 0; r < shared.size();) {
      std::size_t end = r + 1;
      while (end < shared.size() && shared[end].key == shared[r].key)
        ++end;
      if (end - r > 1 || shared[r].layers > 1)
        for (std::size_t k = r; k < end; ++k)
          shared[at++] = shared[k];
      r = end;
    }
    shared.erase_till_end(shared.begin() + std::ptrdiff_t(at));
  }
  if constexpr (tf::core::verification) {
    const std::size_t rows = std::min(std::size_t(owner._corners.size()),
                                      owner._protected_slots.size());
    bool owned = false;
    for (std::size_t cell = 0; cell < rows && !owned; ++cell) {
      if (owner._states[cell] == tet_state::dead)
        continue;
      for (std::size_t slot = 0; slot < 4; ++slot) {
        if (!tet_facet_is_protected(owner, Index(cell), slot))
          continue;
        const Index who = owner._owners[4 * cell + slot];
        owned = owned || (who >= 0 && is_retired(who)) ||
                (who == Owner::shared_owner &&
                 single(tet_facet_key(owner._corners[cell], slot)) !=
                     Owner::shared_owner);
      }
    }
    for (const auto &row : shared)
      owned = owned || is_retired(row.face);
    tf::core::verify(!owned, "no protected facet is owned by a retired face, "
                             "and every stack holds two owners or two layers");
  }
}

} // namespace tf::topology::cdt::dt3

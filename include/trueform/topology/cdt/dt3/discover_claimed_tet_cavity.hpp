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
#include "./claim_tet_cell.hpp"
#include "./close_tet_cavity_links.hpp"
#include "./open_tet_cavity_links.hpp"
#include "./state_tet_cavity_facet.hpp"
#include "./tet_cavity_sides.hpp"
#include "./tet_claim_scratch.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include "./tet_conflicts.hpp"
#include "./tet_facets.hpp"
#include "./tet_site_position.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

template <typename Owner, typename Index>
auto discover_claimed_tet_cavity(const Owner &owner,
                                 tet_claim_workspace<Index> &work,
                                 std::size_t worker, Index site, Index seed,
                                 tet_claim_scratch<Index> &scratch) -> bool {
  auto &cavity = scratch.cavity;
  auto &entered = scratch.entered;
  auto &boundary = scratch.boundary;
  cavity.clear();
  entered.clear();
  boundary.clear();
  auto &links = work.links[worker];
  open_tet_cavity_links(links, tet_site_count(owner));
  cavity.push_back(seed);
  entered.push_back(std::uint8_t(tet_cavity_seed_entrance));
  work.ownership[std::size_t(seed)].store(std::uint32_t(worker) |
                                              tet_classified | tet_conflict,
                                          std::memory_order_relaxed);
  // The side a cell was entered through is its cavity's own and is not asked.
  for (std::size_t head = 0; head < cavity.size(); ++head) {
    const auto cell = cavity[head];
    const std::size_t entrance = entered[head];
    const auto *asked = tet_cavity_sides[entrance];
    const std::size_t n_asked =
        3 + std::size_t(entrance == tet_cavity_seed_entrance);
    for (std::size_t k = 0; k < n_asked; ++k) {
      const auto side = owner._neighbors[std::size_t(cell)][asked[k]];
      // A side's claim is not listed: classified, it stands in the cavity or
      // on the boundary, and those two lists are what an attempt lets go.
      tet_claim_word word =
          work.ownership[std::size_t(side)].load(std::memory_order_relaxed);
      if ((word & tet_worker_mask) != worker) {
        if (!acquire_tet_cell(work, worker, side))
          return false;
        word = tet_claim_word(worker);
      }
      const auto sides = owner._neighbors[std::size_t(side)];
      const std::size_t mate = std::size_t(sides[1] == cell) +
                               2 * std::size_t(sides[2] == cell) +
                               3 * std::size_t(sides[3] == cell);
      if (!(word & tet_classified)) {
        const int inside = tet_conflicts(owner, side, site, [&](Index held) {
          tet_claim_word held_word;
          return claim_tet_cell(work, worker, scratch, held, held_word);
        });
        if (inside < 0) {
          scratch.held.push_back(side);
          return false;
        }
        word = std::uint32_t(worker) | tet_classified |
               (inside ? tet_conflict : 0U);
        work.ownership[std::size_t(side)].store(word,
                                                std::memory_order_relaxed);
        if (inside) {
          cavity.push_back(side);
          entered.push_back(std::uint8_t(mate));
        }
      }
      if (!(word & tet_conflict)) {
        boundary.push_back(Index(4 * std::size_t(side) + mate));
        state_tet_cavity_facet(links, tet_facet(owner._corners[side], mate));
      }
    }
  }
  close_tet_cavity_links(links);
  return true;
}

} // namespace tf::topology::cdt::dt3

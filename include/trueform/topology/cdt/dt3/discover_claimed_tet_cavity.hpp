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
  scratch.cavity.clear();
  scratch.boundary.clear();
  auto &links = work.links[worker];
  open_tet_cavity_links(links, tet_site_count(owner));
  scratch.cavity.push_back(seed);
  work.ownership[std::size_t(seed)].store(std::uint32_t(worker) |
                                              tet_classified | tet_conflict,
                                          std::memory_order_relaxed);
  for (std::size_t head = 0; head < scratch.cavity.size(); ++head) {
    const auto cell = scratch.cavity[head];
    for (std::size_t slot = 0; slot < 4; ++slot) {
      const auto side = owner._neighbors[std::size_t(cell)][slot];
      tet_claim_word word;
      if (!claim_tet_cell(work, worker, scratch, side, word))
        return false;
      if (!(word & tet_classified)) {
        const int inside = tet_conflicts(owner, side, site, [&](Index mate) {
          tet_claim_word mate_word;
          return claim_tet_cell(work, worker, scratch, mate, mate_word);
        });
        if (inside < 0)
          return false;
        word = std::uint32_t(worker) | tet_classified |
               (inside ? tet_conflict : 0U);
        work.ownership[std::size_t(side)].store(word,
                                                std::memory_order_relaxed);
        if (inside)
          scratch.cavity.push_back(side);
      }
      if (!(word & tet_conflict)) {
        const auto sides = owner._neighbors[std::size_t(side)];
        const std::size_t mate = std::size_t(sides[1] == cell) +
                                 2 * std::size_t(sides[2] == cell) +
                                 3 * std::size_t(sides[3] == cell);
        scratch.boundary.push_back(Index(4 * std::size_t(side) + mate));
        state_tet_cavity_facet(links,
                               tet_facet(owner._corners[side], mate));
      }
    }
  }
  close_tet_cavity_links(links);
  return true;
}

} // namespace tf::topology::cdt::dt3

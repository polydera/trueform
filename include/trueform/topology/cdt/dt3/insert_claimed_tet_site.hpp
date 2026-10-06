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
#include "./discover_claimed_tet_cavity.hpp"
#include "./locate_claimed_tet.hpp"
#include "./release_tet_cells.hpp"
#include "./take_claimed_tet_rows.hpp"
#include "./tet_claim_scratch.hpp"
#include "./tet_claim_word.hpp"
#include "./tet_claim_workspace.hpp"
#include "./write_tet_cavity.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

enum class tet_claim_result { inserted, interference, capacity };
template <typename Owner, typename Index>
auto insert_claimed_tet_site(Owner &owner, tet_claim_workspace<Index> &work,
                             std::size_t worker, Index site, Index &hint,
                             tet_claim_scratch<Index> &scratch)
    -> tet_claim_result {
  scratch.held.clear();
  scratch.rows.clear();
  scratch.interfering = -1;
  scratch.interfering_cell = -1;
  const auto seed =
      locate_claimed_tet(owner, work, worker, site, hint, scratch);
  if (seed < 0)
    return tet_claim_result::interference;
  const bool discovered =
      discover_claimed_tet_cavity(owner, work, worker, site, seed, scratch);
  if (!discovered) {
    release_tet_cells(work, worker, scratch.held);
    return tet_claim_result::interference;
  }
  auto &stream = work.streams[worker];
  if (scratch.boundary.size() > stream.free_count + scratch.cavity.size()) {
    hint = seed;
    release_tet_cells(work, worker, scratch.held);
    return tet_claim_result::capacity;
  }
  take_claimed_tet_rows(work, worker, scratch);
  write_tet_cavity(owner, site, scratch.rows, scratch.boundary, scratch.sides,
                   scratch.edges);
  hint = scratch.rows[0];
  // Releasing the neighbors before the rows can make another stream retry,
  // but never shows it a row still being written.
  release_tet_cells(work, worker, scratch.held);
  for (auto cell : scratch.rows)
    work.ownership[std::size_t(cell)].store(tet_unowned,
                                            std::memory_order_release);
  return tet_claim_result::inserted;
}

} // namespace tf::topology::cdt::dt3

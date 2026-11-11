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
#include "./keep_taken_tet_edges.hpp"
#include "./locate_claimed_tet.hpp"
#include "./record_tet_killed_edges.hpp"
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
  const auto seed =
      locate_claimed_tet(owner, work, worker, site, hint, scratch);
  if (seed < 0)
    return tet_claim_result::interference;
  const bool discovered =
      discover_claimed_tet_cavity(owner, work, worker, site, seed, scratch);
  // An attempt holds the walk's cell and the hull mates it listed, its
  // cavity and its boundary; a boundary cell stands there once per facet, so
  // it is let go only while it still names this stream.
  const auto release_boundary = [&work, worker, &scratch] {
    for (auto ticket : scratch.boundary) {
      auto &word = work.ownership[std::size_t(ticket) / 4];
      const auto held = word.load(std::memory_order_relaxed);
      if ((held & tet_worker_mask) == worker && !(held & tet_conflict))
        word.store(tet_unowned, std::memory_order_release);
    }
  };
  const auto give_back = [&work, worker, &scratch, &release_boundary] {
    release_tet_cells(work, worker, scratch.held, 0);
    release_boundary();
    release_tet_cells(work, worker, scratch.cavity, 0);
  };
  if (!discovered) {
    give_back();
    return tet_claim_result::interference;
  }
  auto &stream = work.streams[worker];
  if (scratch.boundary.size() > stream.free_count + scratch.cavity.size()) {
    hint = seed;
    give_back();
    return tet_claim_result::capacity;
  }
  const std::size_t recorded = work.killed[worker].size();
  if (owner._record_killed)
    record_tet_killed_edges(owner, scratch.cavity, work.killed[worker]);
  take_claimed_tet_rows(owner, work, worker, scratch);
  stream.dead_delta += static_cast<long long>(scratch.cavity.size()) -
                       static_cast<long long>(scratch.rows.size());
  write_tet_cavity(owner, site, scratch.rows, scratch.boundary,
                   work.links[worker]);
  if (owner._record_killed)
    keep_taken_tet_edges(work.killed[worker], recorded, work.links[worker]);
  hint = scratch.rows[0];
  // Releasing the neighbors before the rows can make another stream's
  // attempt collide, but never shows it a row still being written.
  release_tet_cells(work, worker, scratch.held, tet_conflict);
  release_boundary();
  for (auto cell : scratch.rows)
    work.ownership[std::size_t(cell)].store(tet_unowned,
                                            std::memory_order_release);
  return tet_claim_result::inserted;
}

} // namespace tf::topology::cdt::dt3

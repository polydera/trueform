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
#include "../../../../../core/algorithm/parallel_for_each.hpp"
#include "../../../../../core/buffer.hpp"
#include "../../../../../core/config/verify.hpp"
#include "../../../../../core/range.hpp"
#include "../../../../../core/views/sequence_range.hpp"
#include "../../tet_execution_tuning.hpp"
#include "../../tet_flip_scratch.hpp"
#include "./bucket_tet_due_rows.hpp"
#include "./lower_tet_round_site_cells.hpp"
#include "./merge_tet_round_states.hpp"
#include "./reserve_tet_region_pools.hpp"
#include "./tet_round_workspace.hpp"
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// ONE ROUND OF THE REGIONS, and the one driver of every phase that walks
/// rows — the segment rounds, the restoration's Lawson pass and its ring
/// removals, the face phase's holds and its uncovered walk: the due work
/// bucketed by the region of its anchor, a pool per region, the regions
/// BESIDE EACH OTHER, and what the admission stopped left for the caller to
/// run in task order on the result.
///
/// `rows_per_item` is the rows one item's work can make, which sizes the
/// pools — zero where the items make none. `patch_of(at)` gives the region
/// of the due item at position `at`;
/// `task(scratch, at)` is the phase's own work on it, which obtains every
/// row through `admit_tet_row(owner, scratch.round, cell)` and leaves the item
/// where it stands the moment the admission stops it. `left` comes back
/// holding the positions the regions did not finish, in the due order, and
/// is the caller's frontier.
///
/// Why the regions are independent: a task writes only rows it was admitted
/// (the mark), it takes fresh rows only from its own pool, the tables the
/// complex owns in common are its own until the barrier
/// (@ref tet_round_state) and are merged in region order
/// (@ref merge_tet_round_states), the site-cell lane is relaxed atomics and
/// is lowered after the barrier (@ref lower_tet_round_site_cells), and the
/// hint is one scalar the round restores. So the product is a function of
/// the due order, the partition and the ladder — the same at any worker
/// count.
///
/// The partition a caller hands over where the mark is already stated: the
/// regions as they stand, named by whoever stated them.
struct tet_standing_partition {
  std::size_t regions = 0;
  auto operator()() const -> std::size_t { return regions; }
};

/// THE PARTITION IS STATED BY THE FIRST ROUND THAT MEETS THE THRESHOLD:
/// `partition()` states the mark once per call and answers the regions it
/// named, every later round reading what stands — so a mesh no round of
/// which meets @ref tet_execution_tuning::region_round_rows states nothing
/// and pays exactly nothing.
///
/// Returns false, having done nothing, where the partition names fewer than
/// two regions or the round is too small to divide; the caller then runs its
/// whole due list itself.
template <typename Owner, typename Index, typename Partition,
          typename PatchOf, typename Task>
auto run_tet_region_round(Owner &owner, const Partition &partition,
                          const tf::buffer<Index> &patches, std::size_t due,
                          std::size_t rows_per_item, const PatchOf &patch_of,
                          const tet_flip_scratch<Index> &like,
                          tet_round_workspace<Index> &work, const Task &task,
                          tf::buffer<Index> &left,
                          tet_flip_scratch<Index> &into) -> bool {
  if (due < tet_execution_tuning::region_round_rows)
    return false;
  const std::size_t n_regions = partition();
  if (n_regions < 2 || !patches.size())
    return false;
  bucket_tet_due_rows(due, patch_of, n_regions, work.cursor, work.offsets,
                      work.order);
  reserve_tet_region_pools(owner, work.offsets, n_regions, rows_per_item,
                           work.pools);
  work.done.allocate_and_initialize(due, char(0));
  work.states.resize(n_regions);
  work.bars.allocate(n_regions);
  struct local_t {
    tet_flip_scratch<Index> scratch;
  };
  const Index hint = owner._hint;
  auto sites = tf::make_range(patches.begin(), patches.end());
  tf::parallel_for_each(
      tf::make_sequence_range(n_regions),
      [&work, &task, &like, sites, rows_per_item](std::size_t p,
                                                  local_t &local) {
        auto &own = local.scratch;
        own.arena.clear();
        own.collected.clear();
        own.touched.clear();
        own.level_limit = like.level_limit;
        own.auto_level = like.auto_level;
        own.star_limit = like.star_limit;
        own.walk_passes = like.walk_passes;
        own.walk_capped = 0;
        own.hull_barred = 0;
        own.round.clear();
        own.round.rows = work.pools[p];
        own.round.sites = sites;
        for (std::size_t i = std::size_t(work.offsets[p]);
             i < std::size_t(work.offsets[p + 1]); ++i) {
          if (rows_per_item &&
              own.round.rows.end - own.round.rows.next < rows_per_item * 2)
            break;
          const Index at = work.order[i];
          open_tet_task(own.round, Index(p));
          const bool through = task(own, at);
          if (through && !own.round.stopped)
            work.done[std::size_t(at)] = 1;
        }
        work.states[p] = std::move(own.round);
        own.round.rows = {};
        work.bars[p] = {own.walk_capped, own.hull_barred};
      },
      local_t{});
  owner._hint = hint;
  for (std::size_t p = 0; p < n_regions; ++p) {
    into.walk_capped += work.bars[p][0];
    into.hull_barred += work.bars[p][1];
  }
  if constexpr (tf::core::verification)
    for (std::size_t p = 0; p < n_regions; ++p)
      for (auto row : work.states[p].written)
        tf::core::verify(tet_row_admits(patches, owner, row, Index(p)),
                         "a region writes only rows its own patch names");
  merge_tet_round_states(owner, work.pools, work.states);
  lower_tet_round_site_cells(owner, work.states);
  for (std::size_t p = 0; p < n_regions; ++p) {
    work.states[p].clear();
    if constexpr (tf::core::verification)
      tf::core::verify(!work.states[p].written.size() &&
                           !work.states[p].killed.size() &&
                           !work.states[p].open(),
                       "a round leaves every region's state merged and spent");
  }
  left.clear();
  for (std::size_t at = 0; at < due; ++at)
    if (!work.done[at])
      left.push_back(Index(at));
  return true;
}

} // namespace tf::topology::cdt::dt3

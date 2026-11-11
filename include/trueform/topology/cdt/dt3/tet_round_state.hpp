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
#include "../../../core/range.hpp"
#include "./recovery/regions/tet_row_admits.hpp"
#include "./recovery/tet_stuck_facet.hpp"
#include "./tet_facet_owner.hpp"
#include "./tet_row_pool.hpp"
#include <array>
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// What one region's own work in a round writes that the complex owns in
/// common: the rows it took (@ref tet_row_pool), the rows it wrote, the
/// edges it killed, the facets it shared (@ref share_tet_facet) and its
/// share of the counts, the held facets among them. A round reserves one of
/// these per region, the region's flips write their own, and the round
/// merges them in region order after its barrier — so no two regions write
/// one word and the merge is the one producer of each table. A closed state
/// is every serial round: the flips write the complex directly.
///
/// THE ADMISSION rides here because the flips already carry it: `sites` is
/// the round's view of the patch of every site (@ref tet_patch_tables), from
/// which a row's own region is READ, never stored — a row belongs to the
/// patch of its first corner (@ref tet_row_admits) — `patch` is the region
/// whose task is in flight, and `stopped` says the task reached a row of
/// another region and stands where it is. Nothing of the admission is a
/// table over the rows, so nothing of it is marked at a birth or carried
/// through a compaction. A task's lifetime is one `patch` and one
/// `stopped` — the round opens them per task (@ref open_tet_task) — and the
/// site patches' is the round.
///
/// A TASK ON A ROUND READS NO SCRATCH BUT ITS REGION'S. The admission guards
/// ROWS, and a lane is no row: a stamp generation two regions share hands
/// each other's marks and a star comes back partial; a lane that grows under
/// another region's writes moves it. So every table a task touches is in its
/// region's own state (@ref tet_flip_scratch), is a per-row lane of the
/// complex sized before the regions run (@ref reserve_tet_region_pools), or
/// is read and never written — the site patches are read-only for the round
/// — and a table that is none of those belongs to the frontier alone.
template <typename Index> struct tet_round_state {
  tet_row_pool rows;
  tf::range<const Index *, tf::dynamic_size> sites;
  Index patch = Index(-1);
  bool stopped = false;
  tf::buffer<Index> written;
  tf::buffer<tet_facet_owner<Index>> shared;
  std::size_t held = 0;
  tf::buffer<std::array<Index, 2>> killed;
  std::size_t flips = 0;
  std::size_t edge_flips = 0;
  std::size_t quad_flips = 0;
  std::size_t dead = 0;
  tf::buffer<tet_stuck_facet<Index>> stuck;

  auto open() const -> bool { return rows.open(); }

  /// Forget what the last round wrote, keeping the capacity.
  auto clear() -> void {
    rows = {};
    patch = Index(-1);
    stopped = false;
    written.clear();
    shared.clear();
    held = 0;
    killed.clear();
    flips = 0;
    edge_flips = 0;
    quad_flips = 0;
    dead = 0;
    stuck.clear();
  }
};

/// Open a task of `round` on the region `patch`: nothing of the last task
/// stands.
template <typename Index>
auto open_tet_task(tet_round_state<Index> &round, Index patch) -> void {
  round.patch = patch;
  round.stopped = false;
}

/// THE ONE READING: whether the patch of `site` is the task's. A row's own
/// region is the patch of its first corner, so this answers for the row
/// standing on it and for every row a flip will MAKE on it, whose corners
/// the kernel knows from the ring before it writes any of them.
template <typename Index>
auto admit_tet_site_patch(tet_round_state<Index> &round, Index site) -> bool {
  if (!round.open())
    return true;
  if (site >= 0 && std::size_t(site) < round.sites.size() &&
      round.sites[std::size_t(site)] == round.patch)
    return true;
  round.stopped = true;
  return false;
}

/// THE ADMISSION: whether the task in flight may take row `cell`, which it
/// obtained through a link. A row of another region stops the task where it
/// stands, and the stop is a deferral — the task has made only its own
/// region's flips, so the complex is valid and the task is simply not
/// finished. Outside a round every row is admitted.
///
/// THE ADMISSION'S UNIT IS THE FLIP'S WHOLE WRITE SET, the undo's included,
/// and it is known before the flip: the ring's cells as they stand and the
/// mates across their facets, which is where an undo reaches
/// (@ref admit_tet_flip_closure). A flip any row of which is foreign is
/// refused up front — the task stops there with nothing written and so
/// nothing to undo — and a region never writes a row another region's mark
/// names.
template <typename Owner, typename Index>
auto admit_tet_row(const Owner &owner, tet_round_state<Index> &round,
                   Index cell) -> bool {
  if (!round.open())
    return true;
  if (cell < 0)
    return true;
  return admit_tet_site_patch(round, owner._corners[std::size_t(cell)][0]);
}

} // namespace tf::topology::cdt::dt3

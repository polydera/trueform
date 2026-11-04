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
#include "../../../core/none.hpp"
#include "../../tetrahedralization_refusal.hpp"
#include "./clear_tet_claim_workspace.hpp"
#include "./clear_tetrahedralization_cells.hpp"
#include "./compact_tet_dead_rows.hpp"
#include "./compact_tetrahedralization.hpp"
#include "./finish_tet_claims.hpp"
#include "./insert_tet_claim_band.hpp"
#include "./insert_tetrahedralization_site.hpp"
#include "./order_appended_tet_sites.hpp"
#include "./prepare_appended_tet_sites.hpp"
#include "./reserve_tet_claim_pool.hpp"
#include "./tet_execution_tuning.hpp"
#include "./tet_insertion.hpp"
#include "./tet_site_position.hpp"
#include "./withdraw_tet_sites.hpp"
#include <algorithm>
#include <cstddef>
#include <tbb/task_arena.h>
#include <type_traits>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// A protected lane appends serially in the canonical order under either
/// policy, so whether a site wraps a protected facet, and is withdrawn, is
/// the same at any worker count, and compacts only by
/// @ref compact_tet_dead_rows. So does an append given `keep`, which reads
/// the owner just before each site `i` of `sites` would be inserted and
/// withdraws the site when it answers false. A `Published` append leaves
/// the complex in canonical order; any other leaves its dead rows to the
/// insertions' own compaction threshold, so its rows are neither canonical
/// nor all live, and its reader reads no row order (@ref state_tet_site_cells reads only live rows),
/// enters no dead row, and publishes once.
template <bool Parallel, bool Published = true, typename Owner, typename Sites,
          typename Work, typename Keep = tf::none_t>
auto append_tetrahedralization_sites(Owner &owner, const Sites &sites,
                                     Work &work, const Keep &keep = {})
    -> bool {
  using Tuning = tet_execution_tuning;
  constexpr bool keeps = !std::is_same_v<Keep, tf::none_t>;
  if (owner._refusal != tf::tetrahedralization_refusal::none ||
      !owner._n_finite)
    return false;
  const auto count = sites.size();
  if (!count)
    return true;
  const std::size_t first = tet_site_count(owner);
  const std::size_t input = owner._index_map.f().size();
  if (!prepare_appended_tet_sites<Parallel>(owner, sites))
    return false;
  if constexpr (Parallel && !keeps) {
    const auto workers = std::size_t(tbb::this_task_arena::max_concurrency());
    if (count >= Tuning::parallel_append_sites &&
        workers >= Tuning::minimum_workers && owner._n_protected == 0 &&
        owner._delaunay) {
      clear_tet_claim_workspace(work);
      reserve_tet_claim_pool(owner, work, workers, count);
      insert_tet_claim_band(owner, work, 0, count);
      return finish_tet_claims(owner, work);
    }
  }
  order_appended_tet_sites(owner, count);
  owner._withdrawn.clear();
  for (auto site : owner._order) {
    if constexpr (keeps)
      if (!keep(std::as_const(owner), std::size_t(site) - first)) {
        owner._withdrawn.push_back(site);
        continue;
      }
    const auto inserted = insert_tetrahedralization_site(owner, site);
    if (inserted == tet_insertion::wrapped) {
      owner._withdrawn.push_back(site);
      continue;
    }
    if (inserted == tet_insertion::capacity) {
      clear_tetrahedralization_cells(owner);
      owner._refusal = tf::tetrahedralization_refusal::index_capacity;
      return false;
    }
  }
  if (owner._n_protected)
    compact_tet_dead_rows<Parallel>(owner);
  else if (Published)
    compact_tetrahedralization<Parallel, true>(owner);
  if (owner._withdrawn.size()) {
    std::sort(owner._withdrawn.begin(), owner._withdrawn.end());
    withdraw_tet_sites(owner, first, input);
  }
  return true;
}

} // namespace tf::topology::cdt::dt3

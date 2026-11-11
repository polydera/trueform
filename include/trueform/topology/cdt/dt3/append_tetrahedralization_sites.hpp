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
#include "../../../core/config/parallel_workers.hpp"
#include "../../../core/none.hpp"
#include "../../tetrahedralization_refusal.hpp"
#include "./clear_tet_claim_workspace.hpp"
#include "./clear_tetrahedralization_cells.hpp"
#include "./compact_tet_dead_rows.hpp"
#include "./finish_tet_claims.hpp"
#include "./insert_tet_claim_band.hpp"
#include "./insert_tetrahedralization_site.hpp"
#include "./open_tet_claim_streams.hpp"
#include "./order_appended_tet_sites.hpp"
#include "./prepare_appended_tet_sites.hpp"
#include "./tet_execution_tuning.hpp"
#include "./tet_insertion.hpp"
#include "./tet_protected_count.hpp"
#include "./tet_site_position.hpp"
#include "./tet_split_route.hpp"
#include "./withdraw_tet_sites.hpp"
#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// A protected lane appends serially in the canonical order under either
/// policy, so whether a site wraps a protected facet, and is withdrawn, is the
/// same at any worker count. So does a complex recording its written rows or
/// its site cells: those lanes are the owner's own and every stream would write
/// them together, where a stream's killed edges are its own and join the owner
/// at the barrier. So does an append given `keep`, which reads the owner just
/// before each site `i` of `sites` would be inserted and withdraws the site
/// when it answers false. An append given the split `route`
/// (@ref tet_split_route) places every site by the split of the simplex it lies
/// in, serially, and wraps no facet. An append costs its own rows: it leaves
/// its dead rows to the owner's compaction threshold
/// (@ref compact_tet_dead_rows), so its rows are neither canonical nor all
/// live, and its reader reads no row order (@ref state_tet_site_cells reads
/// only live rows), enters no dead row, and publishes once.
template <bool Parallel, typename Owner, typename Sites, typename Work,
          typename Keep = tf::none_t, typename Route = tf::none_t>
auto append_tetrahedralization_sites(Owner &owner, const Sites &sites,
                                     Work &work, const Keep &keep = {},
                                     Route route = {}) -> bool {
  using Tuning = tet_execution_tuning;
  constexpr bool keeps = !std::is_same_v<Keep, tf::none_t>;
  constexpr bool splits = std::is_same_v<Route, tet_split_route_t>;
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
  order_appended_tet_sites(owner, count);
  if constexpr (Parallel && !keeps && !splits) {
    const auto workers = tf::core::parallel_workers();
    if (count >= Tuning::parallel_append_sites &&
        workers >= Tuning::minimum_workers && tet_protected_count(owner) == 0 &&
        owner._delaunay && !owner._record_written &&
        !owner._record_site_cells) {
      clear_tet_claim_workspace(work);
      open_tet_claim_streams(owner, work, workers, first, count);
      insert_tet_claim_band(owner, work, 0, count);
      return finish_tet_claims(owner, work);
    }
  }
  owner._withdrawn.clear();
  for (auto site : owner._order) {
    if constexpr (keeps)
      if (!keep(std::as_const(owner), std::size_t(site) - first)) {
        owner._withdrawn.push_back(site);
        continue;
      }
    const auto inserted = insert_tetrahedralization_site(owner, site, route);
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
  compact_tet_dead_rows<Parallel>(owner);
  if (owner._withdrawn.size()) {
    std::sort(owner._withdrawn.begin(), owner._withdrawn.end());
    withdraw_tet_sites(owner, first, input);
  }
  return true;
}

} // namespace tf::topology::cdt::dt3

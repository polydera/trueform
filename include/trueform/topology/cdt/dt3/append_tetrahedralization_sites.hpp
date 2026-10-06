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
#include "../../tetrahedralization_refusal.hpp"
#include "./clear_tet_claim_workspace.hpp"
#include "./clear_tetrahedralization_cells.hpp"
#include "./compact_tetrahedralization.hpp"
#include "./finish_tet_claims.hpp"
#include "./insert_tet_claim_band.hpp"
#include "./insert_tetrahedralization_site.hpp"
#include "./order_tet_site_band.hpp"
#include "./prepare_appended_tet_sites.hpp"
#include "./reserve_tet_claim_pool.hpp"
#include "./tet_cavity_edge_table.hpp"
#include "./tet_execution_tuning.hpp"
#include <cstddef>
#include <tbb/task_arena.h>

namespace tf::topology::cdt::dt3 {

template <bool Parallel, typename Owner, typename Sites, typename Work>
auto append_tetrahedralization_sites(Owner &owner, const Sites &sites,
                                     Work &work) -> bool {
  using Index = typename Owner::index_type;
  using Tuning = tet_execution_tuning;
  if (owner._refusal != tf::tetrahedralization_refusal::none ||
      !owner._n_finite)
    return false;
  const auto count = sites.size();
  if (!count)
    return true;
  if (!prepare_appended_tet_sites<Parallel>(owner, sites))
    return false;
  if constexpr (Parallel) {
    const auto workers = std::size_t(tbb::this_task_arena::max_concurrency());
    if (count >= Tuning::parallel_append_sites &&
        workers >= Tuning::minimum_workers) {
      clear_tet_claim_workspace(work);
      reserve_tet_claim_pool(owner, work, workers, count);
      insert_tet_claim_band(owner, work, 0, count);
      return finish_tet_claims(owner, work);
    }
  }
  order_tet_site_band(owner, 0, count);
  tet_cavity_edge_table<Index> edges{};
  for (auto site : owner._order) {
    if (!insert_tetrahedralization_site(owner, site, edges)) {
      clear_tetrahedralization_cells(owner);
      owner._refusal = tf::tetrahedralization_refusal::index_capacity;
      return false;
    }
  }
  compact_tetrahedralization<Parallel, true>(owner);
  return true;
}

} // namespace tf::topology::cdt::dt3

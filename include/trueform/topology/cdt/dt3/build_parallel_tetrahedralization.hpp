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
#include "./bootstrap_tet_claims.hpp"
#include "./build_tetrahedralization.hpp"
#include "./clear_tet_claim_workspace.hpp"
#include "./clear_tetrahedralization_cells.hpp"
#include "./finish_tet_claims.hpp"
#include "./insert_tet_claim_band.hpp"
#include "./prepare_tet_claim_sites.hpp"
#include "./reserve_tet_claim_pool.hpp"
#include "./tet_claim_workspace.hpp"
#include "./tet_execution_tuning.hpp"
#include <cstddef>
#include <tbb/task_arena.h>

namespace tf::topology::cdt::dt3 {

template <typename Owner, typename Sites>
auto build_parallel_tetrahedralization(
    Owner &owner, const Sites &sites,
    tet_claim_workspace<typename Owner::index_type> &work) -> bool {
  using Tuning = tet_execution_tuning;
  clear_tet_claim_workspace(work);
  const auto workers = std::size_t(tbb::this_task_arena::max_concurrency());
  if (sites.size() < Tuning::parallel_sites ||
      workers < Tuning::minimum_workers)
    return build_tetrahedralization<true>(owner, sites);
  if (!prepare_tet_claim_sites(owner, sites))
    return false;
  auto prefix = owner._order.size();
  while (prefix > Tuning::serial_prefix)
    prefix /= Tuning::band_growth;
  if (!bootstrap_tet_claims(owner, prefix)) {
    clear_tetrahedralization_cells(owner);
    owner._refusal = tf::tetrahedralization_refusal::index_capacity;
    return false;
  }
  reserve_tet_claim_pool(owner, work, workers, owner._order.size() - prefix);
  for (auto begin = prefix; begin < owner._order.size();) {
    const auto end = begin > owner._order.size() / Tuning::band_growth
                         ? owner._order.size()
                         : begin * Tuning::band_growth;
    insert_tet_claim_band(owner, work, begin, end);
    begin = end;
  }
  return finish_tet_claims(owner, work);
}

} // namespace tf::topology::cdt::dt3

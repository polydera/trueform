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
#include "../../tetrahedralization_refusal.hpp"
#include "./bootstrap_tet_claims.hpp"
#include "./build_tetrahedralization.hpp"
#include "./clear_tet_claim_workspace.hpp"
#include "./clear_tetrahedralization_cells.hpp"
#include "./finish_tet_claims.hpp"
#include "./insert_tet_claim_band.hpp"
#include "./open_tet_claim_streams.hpp"
#include "./prepare_tet_claim_sites.hpp"
#include "./tet_claim_workspace.hpp"
#include "./tet_execution_tuning.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

template <typename Owner, typename Sites>
auto build_parallel_tetrahedralization(
    Owner &owner, const Sites &sites,
    tet_claim_workspace<typename Owner::index_type> &work) -> bool {
  using Tuning = tet_execution_tuning;
  clear_tet_claim_workspace(work);
  work.ownership.clear();
  const auto workers = tf::core::parallel_workers();
  // One worker publishes by the serial rank: the parallel one's coarse
  // scatter and bucket copy exist to give workers disjoint writes.
  if (workers == 1)
    return build_tetrahedralization<false>(owner, sites);
  if (sites.size() < Tuning::parallel_sites ||
      workers < Tuning::minimum_workers)
    return build_tetrahedralization<true>(owner, sites);
  if (!prepare_tet_claim_sites(owner, sites))
    return false;
  const auto ordered = owner._order.size();
  auto prefix = ordered;
  while (prefix > Tuning::serial_prefix)
    prefix /= Tuning::band_growth;
  if (!bootstrap_tet_claims(owner, prefix)) {
    clear_tetrahedralization_cells(owner);
    owner._refusal = tf::tetrahedralization_refusal::index_capacity;
    return false;
  }
  open_tet_claim_streams(owner, work, workers,
                         prefix + ordered - owner._order.size(),
                         owner._order.size() - prefix);
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

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
#include "./clear_tetrahedralization_cells.hpp"
#include "./compact_tetrahedralization.hpp"
#include "./insert_tetrahedralization_site.hpp"
#include "./publish_tet_claim_states.hpp"
#include "./tet_cavity_edge_table.hpp"
#include "./tet_claim_workspace.hpp"

namespace tf::topology::cdt::dt3 {

template <typename Owner, typename Index>
auto finish_tet_claims(Owner &owner, tet_claim_workspace<Index> &work) -> bool {
  publish_tet_claim_states(owner, work);
  if (work.tail.size() == 0) {
    compact_tetrahedralization<true, true>(owner);
    return true;
  }
  compact_tetrahedralization<true>(owner);
  tet_cavity_edge_table<Index> edges{};
  for (auto site : work.tail) {
    if (!insert_tetrahedralization_site(owner, site, edges)) {
      clear_tetrahedralization_cells(owner);
      owner._refusal = tf::tetrahedralization_refusal::index_capacity;
      return false;
    }
  }
  compact_tetrahedralization<true, true>(owner);
  return true;
}

} // namespace tf::topology::cdt::dt3

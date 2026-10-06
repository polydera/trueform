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
#include "./clear_tetrahedralization.hpp"
#include "./clear_tetrahedralization_cells.hpp"
#include "./load_tetrahedralization_sites.hpp"
#include "./seed_tetrahedralization.hpp"
#include "./weld_tetrahedralization_sites.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

template <bool Parallel, typename Owner, typename Sites>
auto prepare_tet_sites(Owner &owner, const Sites &sites) -> bool {
  clear_tetrahedralization(owner);
  if (sites.size() > std::size_t(Owner::max_sites)) {
    owner._refusal = tf::tetrahedralization_refusal::index_capacity;
    return false;
  }
  load_tetrahedralization_sites<Parallel>(owner, sites);
  weld_tetrahedralization_sites<Parallel>(owner);
  if (!seed_tetrahedralization(owner)) {
    clear_tetrahedralization_cells(owner);
    owner._refusal = tf::tetrahedralization_refusal::rank_deficient;
    return false;
  }
  return true;
}

} // namespace tf::topology::cdt::dt3

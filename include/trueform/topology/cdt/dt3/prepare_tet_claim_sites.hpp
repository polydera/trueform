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
#include "./prepare_tet_sites.hpp"
#include "./order_tetrahedralization_sites.hpp"

namespace tf::topology::cdt::dt3 {

/// Prepare the sites and state their insertion order.
template <typename Owner, typename Sites>
auto prepare_tet_claim_sites(Owner &owner, const Sites &sites) -> bool {
  if (!prepare_tet_sites<true>(owner, sites))
    return false;
  order_tetrahedralization_sites<true>(owner);
  return true;
}

} // namespace tf::topology::cdt::dt3

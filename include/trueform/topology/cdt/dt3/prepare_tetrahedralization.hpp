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
#include "./order_tetrahedralization_sites.hpp"
#include "./prepare_tet_sites.hpp"

namespace tf::topology::cdt::dt3 {

template <typename Owner, typename Sites>
auto prepare_tetrahedralization(Owner &owner, const Sites &sites) -> bool {
  if (!prepare_tet_sites<false>(owner, sites))
    return false;
  order_tetrahedralization_sites(owner);
  return true;
}

} // namespace tf::topology::cdt::dt3

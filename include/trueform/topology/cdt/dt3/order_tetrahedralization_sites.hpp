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
#include "./encode_tet_site_keys.hpp"
#include "./measure_tet_site_domain.hpp"
#include "./order_tet_site_band.hpp"
#include "./rank_tet_sites.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Rank the sites, then Morton-order each doubling insertion band.
template <typename Owner>
auto order_tetrahedralization_sites(Owner &owner) -> void {
  rank_tet_sites<false>(owner);
  owner._domain = measure_tet_site_domain(owner);
  encode_tet_site_keys(owner, owner._domain);
  for (std::size_t last = owner._sites.size(); last != 0; last >>= 1)
    order_tet_site_band(owner, last >> 1, last);
}

} // namespace tf::topology::cdt::dt3

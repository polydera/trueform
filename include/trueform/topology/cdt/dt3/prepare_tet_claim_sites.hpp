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
#include "./prepare_tet_sites.hpp"
#include "./rank_tet_sites.hpp"

namespace tf::topology::cdt::dt3 {

/// Prepare and rank the sites; each insertion band orders its own sites when
/// it is reached.
template <typename Owner, typename Sites>
auto prepare_tet_claim_sites(Owner &owner, const Sites &sites) -> bool {
  if (!prepare_tet_sites<true>(owner, sites))
    return false;
  rank_tet_sites<true>(owner);
  owner._domain = measure_tet_site_domain<true>(owner);
  encode_tet_site_keys<true>(owner, owner._domain);
  return true;
}

} // namespace tf::topology::cdt::dt3

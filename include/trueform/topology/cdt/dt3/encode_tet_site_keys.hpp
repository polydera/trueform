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
#include "./encode_tet_site_band.hpp"
#include "./tet_site_domain.hpp"

namespace tf::topology::cdt::dt3 {

template <bool Parallel = false, typename Owner>
auto encode_tet_site_keys(
    Owner &owner, const tet_site_domain<typename Owner::coord_type> &domain)
    -> void {
  encode_tet_site_band<Parallel>(owner, 0, owner._sites.size(), domain);
}

} // namespace tf::topology::cdt::dt3

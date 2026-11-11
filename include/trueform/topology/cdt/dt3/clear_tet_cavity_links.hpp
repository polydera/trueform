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
#include "./tet_cavity_links.hpp"
#include <algorithm>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

template <typename Index>
auto clear_tet_cavity_links(tet_cavity_links<Index> &links) -> void {
  links.facets.clear();
  links.across.clear();
  links.n_facets = 0;
  links.records.clear();
  links.local_of_site.clear();
  std::fill(links.table.begin(), links.table.end(), Index(-1));
  links.n_local = 0;
  links.overflowed = false;
  links.pending = false;
}

} // namespace tf::topology::cdt::dt3

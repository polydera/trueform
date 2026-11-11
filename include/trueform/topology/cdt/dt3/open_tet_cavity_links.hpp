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
#include "./scrub_tet_cavity_table.hpp"
#include "./tet_cavity_links.hpp"
#include <cstddef>
#include <cstdint>

namespace tf::topology::cdt::dt3 {

template <typename Index>
auto open_tet_cavity_links(tet_cavity_links<Index> &links, std::size_t n_sites)
    -> void {
  constexpr std::size_t max_local = tet_cavity_links<Index>::max_local;
  if (links.table.size() == 0)
    links.table.allocate_and_initialize(max_local * max_local, Index(-1));
  if (links.pending && !links.overflowed)
    scrub_tet_cavity_table(links, links.n_facets);
  links.n_facets = 0;
  links.records.clear();
  links.n_local = 0;
  links.overflowed = false;
  links.pending = true;
  const std::size_t slots = tet_cavity_stamp_slot(Index(n_sites));
  if (links.local_of_site.size() < slots)
    links.local_of_site.reallocate_and_initialize(slots, std::uint8_t(0));
}

} // namespace tf::topology::cdt::dt3

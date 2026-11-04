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

namespace tf::topology::cdt::dt3 {

/// The boundary has outgrown the table: the facets paired so far leave it
/// empty, and the close pairs the whole boundary from its records.
template <typename Index>
auto overflow_tet_cavity_links(tet_cavity_links<Index> &links,
                               std::size_t n_facets) -> void {
  scrub_tet_cavity_table(links, n_facets);
  links.overflowed = true;
}

} // namespace tf::topology::cdt::dt3

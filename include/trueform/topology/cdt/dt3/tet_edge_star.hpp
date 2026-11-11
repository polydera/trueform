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
#include "../../../core/buffer.hpp"
#include "../../../core/hash_set.hpp"

namespace tf::topology::cdt::dt3 {

/// The cells a search gathers about one site (@ref find_tet_edge,
/// @ref locate_tet_facet), in the order it reaches them: scanned while they
/// are few and entered into `hashed` once the star outgrows the scan
/// (@ref admit_tet_edge_star), so a star of any size costs its own cells.
/// One search's state, carried by whoever runs the search, since searches
/// run beside each other and mark nothing in the complex.
template <typename Index> struct tet_edge_star {
  tf::buffer<Index> cells;
  tf::hash_set<Index> hashed;
};

} // namespace tf::topology::cdt::dt3

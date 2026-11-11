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
#include "../../../../core/buffer.hpp"
#include "./tet_star_stamps.hpp"
#include <array>

namespace tf::topology::cdt::dt3 {

/// The scratch @ref state_tet_site_cells walks on: `stamps` for the stars it
/// gathers, `staged` the site and cell pairs it groups by site and `counts`
/// the blocked counts it scatters them by (@ref scatter_tet_records_by_key).
template <typename Index> struct tet_site_cell_scratch {
  tet_star_stamps stamps;
  tf::buffer<std::array<Index, 2>> staged;
  tf::buffer<Index> counts;
};

} // namespace tf::topology::cdt::dt3

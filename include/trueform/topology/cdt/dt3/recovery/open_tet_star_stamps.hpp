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
#include "./open_tet_stamps.hpp"
#include "./tet_star_stamps.hpp"

namespace tf::topology::cdt::dt3 {

/// Open a generation of `stamps` over the cell rows of `dt`, four marks per
/// row.
template <typename DT>
auto open_tet_star_stamps(const DT &dt, tet_star_stamps &stamps) -> void {
  open_tet_stamps(stamps, 4 * dt.n_tets());
}

} // namespace tf::topology::cdt::dt3

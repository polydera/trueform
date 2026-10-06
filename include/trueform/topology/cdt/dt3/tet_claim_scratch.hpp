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
#include "./tet_cavity_edge_table.hpp"
#include "./tet_side.hpp"

namespace tf::topology::cdt::dt3 {

template <typename Index> struct tet_claim_scratch {
  tet_cavity_edge_table<Index> edges{};
  tf::buffer<Index> held, cavity, boundary, rows;
  tf::buffer<tet_side<Index>> sides;
  int interfering = -1;
  Index interfering_cell = -1;
};

} // namespace tf::topology::cdt::dt3

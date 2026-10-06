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
#include "./tet_cavity_edge_table.hpp"

namespace tf::topology::cdt::dt3 {

template <typename Index>
auto advance_tet_cavity_edges(tet_cavity_edge_table<Index> &table) -> void {
  if (++table.generation == 0) {
    for (auto &slot : table.slots)
      slot.generation = 0;
    table.generation = 1;
  }
}

} // namespace tf::topology::cdt::dt3

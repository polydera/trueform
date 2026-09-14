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

#include "./chart_cell.hpp"
#include "./exact_lane.hpp"

#include "../../../core/buffer.hpp"
#include "../../../core/point.hpp"

#include <array>

namespace tf::exact::door::pool {

/// THE CELL IS THE LINE POOL. One fixed chart cell holds every name whose
/// direction rounds into it; that set is a pure function of the direction
/// alone, so adding geometry can never move an old name out of its cell.
///
/// Each cell ELECTS one of its own: the least canonical exact plane in it.
/// That plane's aligned primitive normal is the cell's LINE, and the
/// original lattice vertex its own supported faces stand on — least by
/// squared norm, then lexicographically — is the point the line passes
/// through. Both come FROM THE GEOMETRY, so nothing the chart invented is
/// ever published.
///
/// THE LINE IS THE CELL'S WHOLE ANSWER. Every member is described by it and
/// no member is asked whether its own direction agrees: what a member offers
/// is the line through its own support vertex, and whether its geometry can
/// reach that plane is the certificate's question alone.
///
/// The election is the one thing here that is NOT invariant under adding
/// geometry: a new least plane in an occupied cell states a new line, so a
/// family that welded before still welds and does so onto a different
/// target.
template <typename Int> struct line_cells {
  tf::buffer<chart_cell> key;
  tf::buffer<int> offsets;
  tf::buffer<int> member;
  tf::buffer<int> cell_of_name;
  tf::buffer<int> sign_of_name;
  tf::buffer<int> elected;
  tf::buffer<std::array<typename exact_lane<Int>::coefficient_type, 3>> line;
  tf::buffer<typename exact_lane<Int>::product_type> square_length;
  tf::buffer<tf::point<Int, 3>> support;
};

} // namespace tf::exact::door::pool

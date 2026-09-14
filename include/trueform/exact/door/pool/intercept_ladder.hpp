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

#include "./exact_lane.hpp"

#include "../../../core/buffer.hpp"

#include <cstddef>

namespace tf::exact::door::pool {

/// EVERY MEMBER OF A CELL, DESCRIBED BY THE CELL'S OWN LINE.
///
/// The line `L` runs through the elected member's support point `q`, and a
/// member plane `(N_i, D_i)` — aligned with the cell's own dominant sign —
/// meets it at
///
///     A_i = D_i - N_i . q,   B_i = N_i . L > 0,   t_i = A_i / B_i,
///
/// so `t_i` is one scalar coordinate every member of the cell shares. The
/// denominator cannot vanish: the cell's angular diameter is under
/// twenty-one degrees, so every member is acute to the line.
///
/// The proposed plane `L . x = L . q + (L . L) t_i` is a DESCRIPTION and is
/// never published or placed onto: its offset is rational and may hold no
/// lattice point at all. What is published is an original exact plane the
/// certificate approved; this ladder only decides who is offered to whom.
///
/// `height` is `t_i sqrt(L . L)`, the physical signed distance along the
/// line, held as a double with the `slack` that bounds its own error. It is
/// a SCREEN and not an authority: every order and every gap it cannot decide
/// inside that slack is handed to the exact comparison.
///
/// `slot_of_name` is the TICKET BACK. A name falls in exactly one cell, so
/// it occupies at most one slot and the ticket is single-valued; `-1` is a
/// name the ladder never placed — one of a silent cell, or one whose aligned
/// plane is not acute to its cell's line. It is what makes "did this run
/// certify that name" one lookup instead of a scan of the run.
template <typename Int> struct intercept_ladder {
  tf::buffer<int> name;
  tf::buffer<int> slot_of_name;
  tf::buffer<int> cell_of_slot;
  tf::buffer<typename exact_lane<Int>::coefficient_type> residual;
  tf::buffer<typename exact_lane<Int>::product_type> reach;
  tf::buffer<double> height;
  tf::buffer<double> slack;
  tf::buffer<int> run_offsets;
};

/// One member's whole place on the line: what the exact comparisons read
/// and what the screen reads, together, so an order or a gap is stated from
/// one value and never from two halves of a record.
template <typename Int> struct ladder_point {
  typename exact_lane<Int>::coefficient_type residual{};
  typename exact_lane<Int>::product_type reach{};
  double height = 0.0;
  double slack = 0.0;
};

template <typename Int>
auto ladder_point_at(const intercept_ladder<Int> &ladder, int slot)
    -> ladder_point<Int> {
  const auto at = std::size_t(slot);
  return ladder_point<Int>{ladder.residual[at], ladder.reach[at],
                           ladder.height[at], ladder.slack[at]};
}

} // namespace tf::exact::door::pool

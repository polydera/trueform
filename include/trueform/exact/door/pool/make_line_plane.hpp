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

#include "../../../core/point.hpp"
#include "../../canonical_plane.hpp"
#include "../../meta.hpp"

namespace tf::exact::door::pool {

/// THE PLANE A MEMBER OFFERS: the cell's elected line, through one original
/// lattice vertex.
///
/// Every member of a cell is described by the elected line, so a member's
/// proposal is not its own equation but the line placed where the member
/// actually stands. Both halves come from the geometry: the direction is a
/// normal some face had, the position a vertex some face stood on.
///
/// It needs no reduction. The line is the elected name's own canonical
/// normal, already primitive and sign-fixed, so the offset against a lattice
/// point completes the canonical quadruple directly — three products and not
/// a gcd.
///
/// A member whose own normal IS the line, asked through its own support
/// vertex, gets its own equation back.
template <typename Int>
auto make_line_plane(const tf::exact::canonical_plane<Int> &elected,
                     const tf::point<Int, 3> &through)
    -> tf::exact::canonical_plane<Int> {
  using T2 = typename tf::exact::meta<Int>::T2;
  return tf::exact::canonical_plane<Int>{
      elected[0], elected[1], elected[2],
      elected[0] * T2(through[0]) + elected[1] * T2(through[1]) +
          elected[2] * T2(through[2])};
}

} // namespace tf::exact::door::pool

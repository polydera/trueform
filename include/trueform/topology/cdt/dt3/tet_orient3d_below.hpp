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
#include "./tet_orient3d_sign.hpp"
#include "../../../exact/vertex.hpp"

namespace tf::topology::cdt::dt3 {

/// The orient3d sign in the reference's reading: positive when `d` lies
/// below the plane of `a`, `b`, `c`, below being the side from which they
/// wind clockwise. A finite cell's corners read negative. The ported
/// routines compare in this sign so each line answers to its reference.
template <typename Int, typename Coord>
auto tet_orient3d_below(const tf::exact::pt3<Coord> &a,
                        const tf::exact::pt3<Coord> &b,
                        const tf::exact::pt3<Coord> &c,
                        const tf::exact::pt3<Coord> &d) -> int {
  return -tet_orient3d_sign<Int>(a, b, c, d);
}

} // namespace tf::topology::cdt::dt3

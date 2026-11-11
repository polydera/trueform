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
#include "../../../../exact/meta.hpp"
#include "../../../../exact/vertex.hpp"
#include "../../../../core/config/verify.hpp"
#include "./tet_orient2d_sign.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Whether `p` lies strictly inside the segment from `a` to `b`, two distinct
/// points on the `Int` lattice carried in `Coord`: on its line when all three
/// projected orientations vanish, then between its ends along an axis the
/// segment spans.
template <typename Int, typename Coord>
auto tet_site_on_segment(const tf::exact::pt3<Coord> &a,
                         const tf::exact::pt3<Coord> &b,
                         const tf::exact::pt3<Coord> &p) -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  for (std::size_t i = 0; i < 3; ++i) {
    const std::size_t j = i == 2 ? 0 : i + 1;
    if (tet_orient2d_sign<Int>(tf::exact::pt2<Coord>{a[i], a[j]},
                               tf::exact::pt2<Coord>{b[i], b[j]},
                               tf::exact::pt2<Coord>{p[i], p[j]}))
      return false;
  }
  if constexpr (tf::core::verification)
    tf::core::verify(a[0] != b[0] || a[1] != b[1] || a[2] != b[2],
                     "a segment's ends are distinct sites");
  std::size_t axis = 0;
  while (axis < 2 && a[axis] == b[axis])
    ++axis;
  const T1 span = T1(b[axis]) - a[axis], along = T1(p[axis]) - a[axis];
  return span > T1(0) ? T1(0) < along && along < span
                      : span < along && along < T1(0);
}

} // namespace tf::topology::cdt::dt3

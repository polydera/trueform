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
#include "./meta.hpp"
#include "./vertex.hpp"
#include <cstddef>

namespace tf::exact {

/// @ingroup exact
/// @brief Is `p` strictly inside the segment `a`-`b`?
///
/// The single producer of betweenness. Assumes `p` is already known to be
/// on the line (an orientation predicate answers that); this answers only
/// where along it. Exact: the projection is compared against the squared
/// length in `T2`, so the answer never depends on a parameter's width.
///
/// Every site that asks the question — the constraint walk, the crossing
/// resolver, and the split registration that patches results back into
/// the pipeline — goes through here. Two tests that disagree let a vertex
/// be "outside" for one and "inside" for another, and a walk then refuses
/// on an obstruction that was never reported.
template <typename Int>
auto is_between_on_segment(const tf::exact::pt2<Int> &a,
                           const tf::exact::pt2<Int> &b,
                           const tf::exact::pt2<Int> &p) -> bool {
  using T2 = typename tf::exact::meta<Int>::T2;
  T2 projection(0);
  T2 length(0);
  for (std::size_t coordinate = 0; coordinate < 2; ++coordinate) {
    const T2 along = T2(b[coordinate]) - T2(a[coordinate]);
    projection += (T2(p[coordinate]) - T2(a[coordinate])) * along;
    length += along * along;
  }
  return T2(0) < projection && projection < length;
}

} // namespace tf::exact

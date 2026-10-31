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
#include "../../core/point.hpp"
#include "../meta.hpp"
#include <cstddef>

namespace tf::exact::door {

/// Whether `v` stands within `band` of `c`: `|v - c|^2 <= band^2`, exactly.
template <typename Int>
auto within_band_of_point(const tf::point<Int, 3> &v,
                          const tf::point<Int, 3> &c, Int band) -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  T2 length(0);
  for (std::size_t k = 0; k < 3; ++k) {
    const T1 delta = T1(v[k]) - T1(c[k]);
    length = length + T2(delta) * T2(delta);
  }
  return length <= T2(band) * T2(band);
}

} // namespace tf::exact::door

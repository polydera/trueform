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

#include <array>
#include <cstddef>
#include <limits>

namespace tf::exact::door {

/// Whether a placement may stand for the vertex it replaces: within the
/// tolerance componentwise, then in length, and inside the lattice the
/// pipeline states its coordinates on.
///
/// Every rank is admitted through here and the last rank leaves the
/// vertex unmoved, so `|placed - original|_inf <= tolerance` holds of
/// the door's whole output by construction; the pair search's `2 T` box
/// growth rests on that.
template <typename Int>
auto admits_placement(const tf::point<Int, 3> &original,
                      const std::array<typename tf::exact::meta<Int>::T1, 3>
                          &placed,
                      Int tolerance) -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  const T1 limit = T1(std::numeric_limits<Int>::max());
  const T1 floor = T1(std::numeric_limits<Int>::min());
  const T1 band = T1(tolerance);
  T2 length(0);
  for (std::size_t k = 0; k < 3; ++k) {
    if (placed[k] > limit || placed[k] < floor)
      return false;
    const T1 delta = placed[k] - T1(original[k]);
    if (delta > band || delta < -band)
      return false;
    length = length + T2(delta) * T2(delta);
  }
  return length <= T2(band) * T2(band);
}

} // namespace tf::exact::door

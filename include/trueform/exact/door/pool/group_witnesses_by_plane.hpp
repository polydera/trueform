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

#include "./pool_records.hpp"

#include "../../../core/algorithm/parallel_fill.hpp"
#include "../../../core/buffer.hpp"

#include <cstddef>

namespace tf::exact::door::pool {

/// How many witnesses each name speaks through: counts and one prefix over
/// the dense name space.
///
/// A name no face speaks for has nothing for a committed plane to realize,
/// so it cannot anchor one — which is what a consumer asks of this. The
/// block itself is the count.
///
/// ADMISSION IS NOT DECIDED HERE. Whether a name passes on the anchor's
/// plane is the certificate's own fact, asked once per (anchor, member) pair
/// by the walk.
template <typename Int>
auto group_witnesses_by_plane(const pool_names<Int> &names,
                              tf::buffer<int> &offsets) -> void {
  const auto count = names.plane.size();
  const auto plane_of = [&names](std::size_t w) {
    return std::size_t(names.witness[w].plane_name);
  };

  offsets.allocate(count + 1);
  tf::parallel_fill(offsets, 0);
  // the count writes one dense slot per witness and the prefix carries a
  // running total, so both are the sweep they are and neither is partitioned
  for (std::size_t w = 0; w < names.witness.size(); ++w)
    ++offsets[plane_of(w)];
  int total = 0;
  for (std::size_t n = 0; n < count; ++n) {
    const int held = offsets[n];
    offsets[n] = total;
    total += held;
  }
  offsets[count] = total;
}

} // namespace tf::exact::door::pool

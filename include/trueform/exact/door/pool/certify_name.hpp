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

#include "./certificate_census.hpp"
#include "./certificate_lane.hpp"
#include "./certify_exact.hpp"
#include "./pool_records.hpp"

#include "../../canonical_plane.hpp"

#include <cstddef>

namespace tf::exact::door::pool {

/// Whether a whole name can join an exact plane: every original vertex of
/// its complete support has a lattice point of that plane within the band.
/// A name is indivisible — one distant occurrence refuses it globally — and
/// a name with no support cannot pass by having nothing to prove.
///
/// The scan carries ONE solved step across itself. A name that is a plane
/// stands wholly at one distance from any other plane of its direction, so
/// its whole support asks the solver the same question: the first vertex
/// buys the answer and the rest compare a residual. What no vertex inherits
/// is its own admission — the sum is the vertex's, so the band and the
/// lattice range are asked of each.
template <typename Int>
auto certify_name(const pool_names<Int> &names, int name,
                  const tf::exact::canonical_plane<Int> &plane, Int tolerance,
                  certificate_lane<Int> &lane, certificate_census &census)
    -> bool {
  const auto from = names.support_offsets[std::size_t(name)];
  const auto to = names.support_offsets[std::size_t(name) + 1];
  if (from == to)
    return false;
  solved_step<Int> step{};
  for (auto k = from; k < to; ++k)
    if (!certify_exact<Int>(plane, names.support[std::size_t(k)], tolerance,
                            lane, step, census)
             .certified)
      return false;
  return true;
}

} // namespace tf::exact::door::pool

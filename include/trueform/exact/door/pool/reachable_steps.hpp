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

#include "../../../core/buffer.hpp"
#include "../../canonical_plane.hpp"
#include "../../meta.hpp"

#include <algorithm>
#include <cstdint>

namespace tf::exact::door::pool {

/// WHICH OFFSETS A LATTICE STEP CAN ACTUALLY REACH along one direction.
///
/// Two planes of a common primitive normal `N` differ by an integer `dD`,
/// and a vertex moves between them only by a lattice step `s` with
/// `N . s == dD`. The band bounds the step, not the offset, so the reachable
/// offsets are the image of a ball:
///
///     S = { N . s : |s|^2 <= T^2 }
///
/// S IS NOT AN INTERVAL AND DOES NOT BECOME ONE. At `T = 1` and
/// `N = (5,8,0)` it is `{0, +-5, +-8}` — an offset two away is unreachable
/// while one five away is a single unit step. Because S is the image of a
/// ball rather than a group, the holes persist at every `T`.
///
/// `enumerated` is false where the ball was too large to walk. Nothing is
/// then stated, every offset counts as reachable, and the certificate
/// remains the one authority on commitment.
template <typename Int> struct reachable_steps {
  tf::buffer<typename tf::exact::meta<Int>::T2> increment;
  bool enumerated = false;
};

/// The ball is worth enumerating while it stays inside this many points,
/// which admits a band up to twelve lattice units. It is an AFFORDABILITY
/// bound and states nothing about where the holes stop; the walk pays it
/// once per cell.
inline constexpr std::int64_t reachable_step_budget = 16384;

/// The reachable offsets of `normal` within `tolerance`, or nothing when the
/// ball is too large to walk.
template <typename Int>
auto make_reachable_steps(const tf::exact::canonical_plane<Int> &normal,
                          Int tolerance) -> reachable_steps<Int> {
  using T2 = typename tf::exact::meta<Int>::T2;

  reachable_steps<Int> at;
  const std::int64_t reach = std::int64_t(tolerance);
  if (reach < 0)
    return at;
  const std::int64_t span = 2 * reach + 1;
  const std::int64_t edge = reachable_step_budget;
  if (span > edge || span * span > edge || span * span * span > edge)
    return at;

  const std::int64_t square = reach * reach;
  for (std::int64_t x = -reach; x <= reach; ++x)
    for (std::int64_t y = -reach; y <= reach; ++y)
      for (std::int64_t z = -reach; z <= reach; ++z) {
        if (x * x + y * y + z * z > square)
          continue;
        at.increment.push_back(normal[0] * T2(x) + normal[1] * T2(y) +
                               normal[2] * T2(z));
      }
  std::sort(at.increment.begin(), at.increment.end());
  at.increment.erase_till_end(
      std::unique(at.increment.begin(), at.increment.end()));
  at.enumerated = true;
  return at;
}

} // namespace tf::exact::door::pool

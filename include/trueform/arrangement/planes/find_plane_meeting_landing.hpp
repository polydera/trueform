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

#include "../../core/buffer.hpp"
#include "../../exact/meta.hpp"
#include "./plane_landing_parameter.hpp"
#include <cstddef>
#include <type_traits>

namespace tf::arrangement {

/// What a crossing of two constraints is stated as. `shared`: the two share
/// an end, so they cross only where the triangulation's own rounding made
/// them, and the meeting states nothing. Otherwise an end of one landing on
/// the other: `point` the prepared point, `constraint` the one it lands on,
/// `parameter` where along it as the plane fed it; `point` is `-1` where no
/// end lands and the meeting is minted.
template <typename Index, typename Param> struct plane_meeting_landing {
  Index point = Index(-1);
  Index constraint = Index(-1);
  Param parameter = Param(0);
  bool shared = false;
};

/// THE MEETING LANDING. A crossing of two constraints whose end is a point
/// this wave minted, and stands within a lattice unit of the other
/// constraint inside its span, is stated at the position the lattice can
/// represent: the end landing on the other constraint, at the parameter
/// @ref plane_landing_parameter states, not a point of its own. Minting it
/// again rounds off both near-collinear lines and crosses the other a few
/// units on, round after round; a landed end is shared, and two constraints
/// sharing an end cannot cross. The needle the two pieces bounded between
/// the meeting and the end is narrower than one lattice unit, so the landing
/// collapses nothing the lattice could state. `first_minted` is the first
/// flat identity the wave minted — every one before it the world named.
template <typename Index, typename Param, typename Points2>
auto find_plane_meeting_landing(const tf::buffer<Index> &cons,
                                const Points2 &pts2,
                                const tf::buffer<Index> &ends,
                                Index first_minted, Index id_a, Index id_b)
    -> plane_meeting_landing<Index, Param> {
  using Int = std::decay_t<decltype(pts2[0][0])>;
  using T2 = typename tf::exact::meta<Int>::T2;
  using T3 = typename tf::exact::meta<Int>::T3;
  const Index pair[2] = {id_a, id_b};
  for (int end_a = 0; end_a < 2; ++end_a)
    for (int end_b = 0; end_b < 2; ++end_b)
      if (cons[2 * std::size_t(id_a) + std::size_t(end_a)] ==
          cons[2 * std::size_t(id_b) + std::size_t(end_b)]) {
        plane_meeting_landing<Index, Param> shared;
        shared.shared = true;
        return shared;
      }
  for (int side = 0; side < 2; ++side) {
    const auto self = std::size_t(pair[side]);
    const auto other = std::size_t(pair[1 - side]);
    const auto a = pts2[std::size_t(cons[2 * other])];
    const auto b = pts2[std::size_t(cons[2 * other + 1])];
    const T2 dx = T2(b[0]) - T2(a[0]);
    const T2 dy = T2(b[1]) - T2(a[1]);
    const T2 length2 = dx * dx + dy * dy;
    for (int end = 0; end < 2; ++end) {
      const auto e = cons[2 * self + std::size_t(end)];
      if (ends[std::size_t(e)] < first_minted)
        continue;
      const auto p = pts2[std::size_t(e)];
      const T3 off =
          T3(dx * (T2(p[1]) - T2(a[1])) - dy * (T2(p[0]) - T2(a[0])));
      if (off * off >= T3(length2))
        continue;
      const auto parameter = plane_landing_parameter<Int>(a, b, p);
      if (parameter == Param(-1))
        continue;
      return {e, Index(other), parameter};
    }
  }
  return {};
}

} // namespace tf::arrangement

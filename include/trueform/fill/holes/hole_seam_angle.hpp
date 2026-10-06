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
#include "./hole_fairing_stencil.hpp"
#include "./hole_metric.hpp"
#include <algorithm>
#include <array>
#include <cstddef>

namespace tf::fill {

/// The largest angle the patch turns through where it meets the host.
///
/// Every rim subedge is one directed side of a patch triangle and the
/// opposite side of the collar triangle across it — the carrying face's own
/// plan — so the seam is read off the stencil alone, and a patch lying flat
/// against its host reads zero. A pair either of whose triangles states no
/// plane is not charged.
template <typename Index>
auto hole_seam_angle(const tf::fill::hole_fairing_stencil<Index> &stencil,
                     tf::buffer<std::array<Index, 3>> &scratch) -> double {
  const auto normal_of = [&stencil](std::size_t triangle) {
    const auto corners = stencil.triangles[triangle];
    return tf::fill::hole_oriented_normal(
        stencil.positions[std::size_t(corners[0])],
        stencil.positions[std::size_t(corners[1])],
        stencil.positions[std::size_t(corners[2])]);
  };

  scratch.clear();
  for (std::size_t triangle = 0; triangle < stencil.triangles.size();
       ++triangle) {
    if (stencil.from_patch[triangle])
      continue;
    const auto corners = stencil.triangles[triangle];
    for (int side = 0; side < 3; ++side)
      scratch.push_back({corners[std::size_t(side)],
                         corners[std::size_t((side + 1) % 3)],
                         Index(triangle)});
  }
  std::sort(scratch.begin(), scratch.end());

  double stated = 0.0;
  for (std::size_t triangle = 0; triangle < stencil.triangles.size();
       ++triangle) {
    if (!stencil.from_patch[triangle])
      continue;
    const auto patch = normal_of(triangle);
    if (!tf::fill::hole_normal_is_usable(patch))
      continue;
    const auto corners = stencil.triangles[triangle];
    for (int side = 0; side < 3; ++side) {
      const std::array<Index, 3> probe{corners[std::size_t((side + 1) % 3)],
                                       corners[std::size_t(side)], Index(0)};
      const auto found = std::lower_bound(
          scratch.begin(), scratch.end(), probe,
          [](const std::array<Index, 3> &x, const std::array<Index, 3> &y) {
            return x[0] != y[0] ? x[0] < y[0] : x[1] < y[1];
          });
      if (found == scratch.end() || (*found)[0] != probe[0] ||
          (*found)[1] != probe[1])
        continue;
      const auto collar = normal_of(std::size_t((*found)[2]));
      if (!tf::fill::hole_normal_is_usable(collar))
        continue;
      stated = std::max(stated, tf::fill::hole_dihedral_angle(patch, collar));
    }
  }
  return stated;
}

} // namespace tf::fill

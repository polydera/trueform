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
#include "../../../exact/meta.hpp"
#include "../../../exact/orient3d.hpp"
#include "./close_seed_tet.hpp"
#include "./write_tet.hpp"
#include <cstddef>

namespace tf::topology::cdt::dt3 {

/// Seed on the first affine basis in canonical site order and close its hull.
/// Returns false when the sites span fewer than three dimensions.
template <typename Owner> auto seed_tetrahedralization(Owner &owner) -> bool {
  using Index = typename Owner::index_type;
  using Int = typename Owner::int_type;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;

  const std::size_t n = owner._sites.size();
  if (n < 4)
    return false;
  const auto &a = owner._sites[0].pt;
  const auto &b = owner._sites[1].pt;
  const T1 ux = T1(b[0]) - a[0], uy = T1(b[1]) - a[1], uz = T1(b[2]) - a[2];

  std::size_t c = 2;
  for (; c < n; ++c) {
    const auto &q = owner._sites[c].pt;
    const T1 vx = T1(q[0]) - a[0], vy = T1(q[1]) - a[1], vz = T1(q[2]) - a[2];
    if (T2(uy) * vz - T2(uz) * vy != T2(0) ||
        T2(uz) * vx - T2(ux) * vz != T2(0) ||
        T2(ux) * vy - T2(uy) * vx != T2(0))
      break;
  }
  if (c == n)
    return false;

  std::size_t d = c + 1;
  T2 volume = T2(0);
  for (; d < n; ++d) {
    volume = tf::exact::orient3d_value_scaled<Int>(a, b, owner._sites[c].pt,
                                                   owner._sites[d].pt);
    if (volume != T2(0))
      break;
  }
  if (d == n)
    return false;

  const Index seed = Index(owner._corners.size());
  owner._corners.reallocate(std::size_t(seed) + 1);
  owner._neighbors.reallocate(std::size_t(seed) + 1);
  owner._states.reallocate(std::size_t(seed) + 1);
  if (volume > T2(0))
    write_tet(owner, seed, Index(0), Index(1), Index(c), Index(d));
  else
    write_tet(owner, seed, Index(0), Index(1), Index(d), Index(c));
  close_seed_tet(owner, seed);
  owner._hint = seed;
  return true;
}

} // namespace tf::topology::cdt::dt3

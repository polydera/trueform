/*
 * Copyright (c) 2025 XLAB
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

#include "../core/buffer.hpp"
#include "../core/point.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <utility>

namespace tf::cut {

namespace detail {

inline constexpr double k_lip = 1.0;  // sizing Lipschitz constant
inline constexpr double k_beta = 1.5; // edge-length term relax factor
inline constexpr double k_size = 1.5; // split when len > k_size * h
inline constexpr std::size_t k_lfs_cap = 2048; // O(n^2) lfs guard

inline auto seg_dist(const std::array<double, 2> &p,
                     const std::array<double, 2> &a,
                     const std::array<double, 2> &b) -> double {
  double dx = b[0] - a[0], dy = b[1] - a[1];
  double l2 = dx * dx + dy * dy;
  double t = 0;
  if (l2 > 0)
    t = std::clamp(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2, 0.0, 1.0);
  double qx = a[0] + t * dx - p[0], qy = a[1] + t * dy - p[1];
  return std::sqrt(qx * qx + qy * qy);
}

/// Sizing field on a closed cycle of 2D points: min adjacent edge length,
/// optionally min distance to non-adjacent segments (lfs proxy), then cyclic
/// Lipschitz smoothing in both directions.
template <typename Pts>
inline auto cycle_sizing(const Pts &pts, bool with_lfs,
                         tf::buffer<double> &h, tf::buffer<double> &elen)
    -> void {
  const std::size_t n = pts.size();
  elen.allocate(n);
  for (std::size_t i = 0; i < n; ++i) {
    const auto &a = pts[i];
    const auto &b = pts[(i + 1) % n];
    elen[i] = std::hypot(b[0] - a[0], b[1] - a[1]);
  }
  h.allocate(n);
  for (std::size_t i = 0; i < n; ++i)
    h[i] = k_beta * std::min(elen[i], elen[(i + n - 1) % n]);
  if (with_lfs && n <= k_lfs_cap) {
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t prev = (i + n - 1) % n;
      for (std::size_t j = 0; j < n; ++j) {
        if (j == i || j == prev)
          continue;
        double d = seg_dist(pts[i], pts[j], pts[(j + 1) % n]);
        if (d > 0)
          h[i] = std::min(h[i], d);
      }
    }
  }
  for (int pass = 0; pass < 2; ++pass) {
    for (std::size_t i = 0; i < n; ++i) {
      std::size_t j = (i + 1) % n;
      h[j] = std::min(h[j], h[i] + k_lip * elen[i]);
    }
    for (std::size_t i = n; i-- > 0;) {
      std::size_t j = (i + n - 1) % n;
      h[j] = std::min(h[j], h[i] + k_lip * elen[j]);
    }
  }
}

/// Recursive dyadic splitting of param interval [pa, pb] (256-unit space);
/// sizing = min(linear interpolation, true lfs at the midpoint via LfsAt).
/// Alignment guards keep every emitted param a multiple of 4 (depth <= 6)
/// even on unioned trees.
template <typename Emit, typename LfsAt>
inline auto dyadic_split(std::uint32_t pa, std::uint32_t pb,
                         const std::array<double, 2> &xa,
                         const std::array<double, 2> &xb, double ha, double hb,
                         Emit &&emit, LfsAt &&lfs_at) -> void {
  if (pb - pa <= 4)
    return;
  std::uint32_t pm = (pa + pb) / 2;
  if (pm & 3)
    return;
  std::array<double, 2> xm{0.5 * (xa[0] + xb[0]), 0.5 * (xa[1] + xb[1])};
  double len = std::hypot(xb[0] - xa[0], xb[1] - xa[1]);
  double hm = std::min(0.5 * (ha + hb), lfs_at(xm));
  if (len <= k_size * hm)
    return;
  emit(pm);
  dyadic_split(pa, pm, xa, xm, ha, hm, emit, lfs_at);
  dyadic_split(pm, pb, xm, xb, hm, hb, emit, lfs_at);
}

/// Lift a 2D point back onto the face plane (int lattice).
template <typename Int>
inline auto lift3(Int x, Int y, int af, int as, const double nd[3],
                  const double ad[3]) -> tf::point<Int, 3> {
  tf::point<Int, 3> p{};
  p[std::size_t(af)] = Int(x);
  p[std::size_t(as)] = Int(y);
  int drop = 3 - af - as;
  double z = ad[std::size_t(drop)];
  if (nd[drop] != 0) {
    double s2 = nd[std::size_t(af)] * (double(x) - ad[std::size_t(af)]) +
                nd[std::size_t(as)] * (double(y) - ad[std::size_t(as)]);
    z = ad[std::size_t(drop)] - s2 / nd[drop];
  }
  p[std::size_t(drop)] = Int(std::llround(z));
  return p;
}

/// param (256-space, multiple of 4) -> (odd num, depth <= 6)
inline auto param_to_rec(std::uint32_t p)
    -> std::pair<std::uint8_t, std::uint8_t> {
  int depth = 8;
  while (depth > 0 && (p & 1) == 0) {
    p >>= 1;
    --depth;
  }
  return {std::uint8_t(p), std::uint8_t(depth)};
}

} // namespace detail

} // namespace tf::cut

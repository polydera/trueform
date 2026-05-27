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

#include "../core/sqrt.hpp"
#include "./int128.hpp"
#include "./int256.hpp"
#include "./meta.hpp"
#include "./vertex.hpp"
#include <cmath>

namespace tf::exact {

namespace detail {

/// Convert an unsigned 128-bit integer to double via 64-bit limb split.
inline auto u128_to_double(tf::exact::uint128 v) -> double {
  std::uint64_t lo = static_cast<std::uint64_t>(v);
  std::uint64_t hi = static_cast<std::uint64_t>(v >> 64);
  return double(hi) * 18446744073709551616.0 + double(lo);
}

inline auto to_double(int v) -> double { return double(v); }
inline auto to_double(long v) -> double { return double(v); }
inline auto to_double(long long v) -> double { return double(v); }
inline auto to_double(tf::exact::int128 v) -> double {
  bool neg = v < 0;
  tf::exact::uint128 u = neg ? tf::exact::uint128(-v) : tf::exact::uint128(v);
  double d = u128_to_double(u);
  return neg ? -d : d;
}
inline auto to_double(tf::exact::int256 v) -> double {
  bool neg = v.is_negative();
  if (neg) v = -v;
  double d_lo = u128_to_double(v.lo());
  double d_hi = u128_to_double(v.hi());
  // 2^128 = (2^64)^2
  constexpr double k_2pow128 =
      18446744073709551616.0 * 18446744073709551616.0;
  double d = d_hi * k_2pow128 + d_lo;
  return neg ? -d : d;
}

/// Convert a double back to a wide integer via 64-bit limb decomposition.
/// Caller applies `ceil` upstream for a conservative upper bound.
template <typename T2> auto double_to_int(double v) -> T2 {
  if (v == 0) return T2(0);
  bool neg = v < 0;
  if (neg) v = -v;
  T2 result = T2(0);
  constexpr double k_2pow64 = 18446744073709551616.0;
  unsigned shift = 0;
  while (v >= 0.5 && shift < 256) {
    std::uint64_t lo = static_cast<std::uint64_t>(std::fmod(v, k_2pow64));
    result = result + (T2(lo) << shift);
    v = std::floor(v / k_2pow64);
    shift += 64;
  }
  return neg ? -result : result;
}

} // namespace detail

/// Volume threshold for orient3d such that `|orient3d(a, b, c, d)| <= bound`
/// iff `d` is within `dist_tol_int` of the plane through (a, b, c).
///
/// `|6V| = |cross(b-a, c-a)| * h`, so bound = `|cross| * dist_tol_int`.
/// `|cross|` via double sqrt, ceil-rounded so the band is never undersized.
template <typename Int>
auto orient3d_distance_bound(const pt3<Int> &a, const pt3<Int> &b,
                             const pt3<Int> &c, double dist_tol_int) ->
    typename meta<Int>::T2 {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 u0 = T1(b[0]) - T1(a[0]);
  T1 u1 = T1(b[1]) - T1(a[1]);
  T1 u2 = T1(b[2]) - T1(a[2]);
  T1 v0 = T1(c[0]) - T1(a[0]);
  T1 v1 = T1(c[1]) - T1(a[1]);
  T1 v2 = T1(c[2]) - T1(a[2]);
  T2 nx = T2(u1) * v2 - T2(u2) * v1;
  T2 ny = T2(u2) * v0 - T2(u0) * v2;
  T2 nz = T2(u0) * v1 - T2(u1) * v0;
  double dx = detail::to_double(nx);
  double dy = detail::to_double(ny);
  double dz = detail::to_double(nz);
  double cross_mag = tf::sqrt(dx * dx + dy * dy + dz * dz);
  return detail::double_to_int<T2>(std::ceil(cross_mag * dist_tol_int));
}

/// Area threshold for orient2d such that `|orient2d(a, b, c)| <= bound`
/// iff `c` is within `dist_tol_int` of the line through (a, b).
/// Bound = `|b - a| * dist_tol_int`.
template <typename Int>
auto orient2d_distance_bound(const pt2<Int> &a, const pt2<Int> &b,
                             double dist_tol_int) ->
    typename meta<Int>::T2 {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 e0 = T1(b[0]) - T1(a[0]);
  T1 e1 = T1(b[1]) - T1(a[1]);
  double e0d = detail::to_double(e0);
  double e1d = detail::to_double(e1);
  double edge_mag = tf::sqrt(e0d * e0d + e1d * e1d);
  return detail::double_to_int<T2>(std::ceil(edge_mag * dist_tol_int));
}

/// 3D-edge-projected variant: `c` (3D) projected onto axes (ax0, ax1) is
/// within `dist_tol_int` of the projected line through (a, b).
template <typename Int>
auto orient2d_distance_bound(const pt3<Int> &a, const pt3<Int> &b, int ax0,
                             int ax1, double dist_tol_int) ->
    typename meta<Int>::T2 {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 e0 = T1(b[ax0]) - T1(a[ax0]);
  T1 e1 = T1(b[ax1]) - T1(a[ax1]);
  double e0d = detail::to_double(e0);
  double e1d = detail::to_double(e1);
  double edge_mag = tf::sqrt(e0d * e0d + e1d * e1d);
  return detail::double_to_int<T2>(std::ceil(edge_mag * dist_tol_int));
}

/// Volume threshold for orient3d on 4 points with skew-line semantics:
/// `|orient3d(a, b, c, d)| <= bound` iff the shortest distance between
/// lines (a, b) and (c, d) is at most `dist_tol_int`.
///
/// Shortest skew-line distance = `|orient3d| / |(b-a) x (d-c)|`, so
/// bound = `|cross(b-a, d-c)| * dist_tol_int`.
template <typename Int>
auto ee_distance_bound(const pt3<Int> &a, const pt3<Int> &b,
                       const pt3<Int> &c, const pt3<Int> &d,
                       double dist_tol_int) -> typename meta<Int>::T2 {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  T1 u0 = T1(b[0]) - T1(a[0]);
  T1 u1 = T1(b[1]) - T1(a[1]);
  T1 u2 = T1(b[2]) - T1(a[2]);
  T1 v0 = T1(d[0]) - T1(c[0]);
  T1 v1 = T1(d[1]) - T1(c[1]);
  T1 v2 = T1(d[2]) - T1(c[2]);
  T2 nx = T2(u1) * v2 - T2(u2) * v1;
  T2 ny = T2(u2) * v0 - T2(u0) * v2;
  T2 nz = T2(u0) * v1 - T2(u1) * v0;
  double dx = detail::to_double(nx);
  double dy = detail::to_double(ny);
  double dz = detail::to_double(nz);
  double cross_mag = tf::sqrt(dx * dx + dy * dy + dz * dz);
  return detail::double_to_int<T2>(std::ceil(cross_mag * dist_tol_int));
}

} // namespace tf::exact

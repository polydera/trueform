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
#include "../../../../exact/meta.hpp"
#include "../../../../exact/orient3d.hpp"
#include "../../../../exact/vertex.hpp"
#include "../tet_site_width.hpp"
#include "./tet_plane_normal.hpp"
#include "./tet_point_in_facet.hpp"
#include "./tet_projected_point.hpp"
#include <array>
#include <cstddef>
#include <utility>

namespace tf::topology::cdt::dt3 {

/// A point of `facet` off its rim near where the edge d e crosses its plane
/// (`od`, `oe` their volumes), on the plane's lattice among the decision's
/// `sites` (@ref tet_decision_sites) `coarse` bits apart: the crossing
/// rounded in the reduced lattice basis, else the point nearest it on the
/// nearest line holding one. The reduced basis is scale-free, so the lattice
/// of a face read `coarse` bits shallower than its sites is the basis
/// stepped `2^coarse` at a time from its first corner. Every point the
/// search states is read on the plane exactly, so a value past the
/// lattice's headroom refuses rather than naming a point off the face.
template <typename Sites, typename Facet>
auto tet_plane_lattice_point(
    const Sites &sites, const Facet &facet, const std::pair<int, int> &axes,
    typename Sites::index_type d, typename Sites::index_type e,
    const typename tf::exact::meta<typename Sites::int_type>::T2 &od,
    const typename tf::exact::meta<typename Sites::int_type>::T2 &oe,
    tf::exact::pt3<typename Sites::coord_type> &out, int coarse) -> bool {
  using Int = typename Sites::int_type;
  using Coord = typename Sites::coord_type;
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  using T3 = typename tf::exact::meta<Int>::T3;
  using v2 = std::array<T2, 3>;
  using p2 = std::array<T2, 2>;
  const auto &plane = facet[0];
  const auto a = sites[std::size_t(plane[0])].pt;
  const auto b = sites[std::size_t(plane[1])].pt;
  const auto c = sites[std::size_t(plane[2])].pt;
  const auto pd = sites[std::size_t(d)].pt;
  const auto pe = sites[std::size_t(e)].pt;
  const auto abs2 = [](const T2 &x) { return x < T2(0) ? -x : x; };
  const auto gcd2 = [&abs2](T2 x, T2 y) {
    x = abs2(x);
    y = abs2(y);
    while (y != T2(0)) {
      const T2 r = x % y;
      x = y;
      y = r;
    }
    return x;
  };
  auto n = tet_plane_normal<Int>(a, b, c);
  const T2 g = gcd2(gcd2(n[0], n[1]), n[2]);
  if (g == T2(0))
    return false;
  for (auto &k : n)
    k = k / g;
  v2 u1{}, u2{};
  if (n[0] == T2(0) && n[1] == T2(0)) {
    u1 = {T2(1), T2(0), T2(0)};
    u2 = {T2(0), T2(1), T2(0)};
  } else {
    const T2 g2 = gcd2(n[0], n[1]);
    u1 = {n[1] / g2, -(n[0] / g2), T2(0)};
    T2 old_r = abs2(n[0]), r = abs2(n[1]);
    T2 old_s(1), s(0), old_t(0), t(1);
    while (r != T2(0)) {
      const T2 q = old_r / r;
      T2 next = old_r - q * r;
      old_r = r;
      r = next;
      next = old_s - q * s;
      old_s = s;
      s = next;
      next = old_t - q * t;
      old_t = t;
      t = next;
    }
    if (n[0] < T2(0))
      old_s = -old_s;
    if (n[1] < T2(0))
      old_t = -old_t;
    u2 = {old_s * n[2], old_t * n[2], -g2};
  }
  const auto dot = [](const v2 &x, const v2 &y) {
    return T3(x[0]) * T3(y[0]) + T3(x[1]) * T3(y[1]) + T3(x[2]) * T3(y[2]);
  };
  const auto floor_div = [](const T3 &p, const T3 &q) {
    T3 quotient = p / q;
    const T3 remainder = p % q;
    if (remainder != T3(0) && ((remainder < T3(0)) != (q < T3(0))))
      quotient = quotient - T3(1);
    return quotient;
  };
  const auto round_div = [&floor_div](const T3 &p, const T3 &q) {
    const T3 two(2);
    return q < T3(0) ? floor_div(two * -p + -q, two * -q)
                     : floor_div(two * p + q, two * q);
  };
  for (;;) {
    if (dot(u1, u1) > dot(u2, u2))
      std::swap(u1, u2);
    const T3 k = round_div(dot(u1, u2), dot(u1, u1));
    if (k == T3(0))
      break;
    for (std::size_t i = 0; i < 3; ++i)
      u2[i] = T2(T3(u2[i]) - k * T3(u1[i]));
  }
  for (std::size_t i = 0; i < 3; ++i) {
    u1[i] = u1[i] * (T2(1) << coarse);
    u2[i] = u2[i] * (T2(1) << coarse);
  }
  const v2 w{u1[1] * u2[2] - u1[2] * u2[1], u1[2] * u2[0] - u1[0] * u2[2],
             u1[0] * u2[1] - u1[1] * u2[0]};
  const T2 den = od - oe;
  std::array<T3, 3> num{};
  for (std::size_t i = 0; i < 3; ++i)
    num[i] =
        T3(T1(pd[i]) - T1(a[i])) * T3(den) + T3(od) * T3(T1(pe[i]) - T1(pd[i]));
  const auto cross3 = [](const std::array<T3, 3> &x, const v2 &y) {
    return std::array<T3, 3>{x[1] * T3(y[2]) - x[2] * T3(y[1]),
                             x[2] * T3(y[0]) - x[0] * T3(y[2]),
                             x[0] * T3(y[1]) - x[1] * T3(y[0])};
  };
  const auto cross3r = [](const v2 &x, const std::array<T3, 3> &y) {
    return std::array<T3, 3>{T3(x[1]) * y[2] - T3(x[2]) * y[1],
                             T3(x[2]) * y[0] - T3(x[0]) * y[2],
                             T3(x[0]) * y[1] - T3(x[1]) * y[0]};
  };
  const auto dot3 = [](const std::array<T3, 3> &x, const v2 &y) {
    return x[0] * T3(y[0]) + x[1] * T3(y[1]) + x[2] * T3(y[2]);
  };
  const T3 reach = T3(1) << tet_site_width<Int, Coord>::storage_bits;
  const auto state = [&a, &b, &c, &u1, &u2, &out, &reach](const T3 &sigma,
                                                          const T3 &tau) {
    for (std::size_t i = 0; i < 3; ++i) {
      const T3 value = T3(T1(a[i])) + sigma * T3(u1[i]) + tau * T3(u2[i]);
      if (!(value > -reach && value < reach))
        return false;
      out[i] = Coord(T1(value));
    }
    return tf::exact::orient3d_value_scaled<Int>(a, b, c, out) == T2(0);
  };
  const auto inside = [&state, &sites, &facet, &axes, &out](const T3 &i,
                                                            const T3 &j) {
    return state(i, j) && tet_point_in_facet(sites, facet, axes, out);
  };
  const T3 scale = T3(den) * dot(w, w);
  const T3 sigma = round_div(dot3(cross3(num, u2), w), scale);
  const T3 rise = dot3(cross3r(u1, num), w);
  const T3 tau = round_div(rise, scale);
  if (inside(sigma, tau))
    return true;
  const auto project = [&axes](const v2 &u) {
    return p2{u[std::size_t(axes.first)], u[std::size_t(axes.second)]};
  };
  const auto cross2 = [](const p2 &x, const p2 &y) {
    return x[0] * y[1] - x[1] * y[0];
  };
  const auto from = [](const tf::exact::pt2<Coord> &p,
                       const tf::exact::pt2<Coord> &q) {
    return p2{T2(T1(q[0]) - T1(p[0])), T2(T1(q[1]) - T1(p[1]))};
  };
  const auto origin = tet_projected_point(a, axes);
  const auto corner = [&sites, &axes](std::size_t v) {
    return tet_projected_point(sites[v].pt, axes);
  };
  const p2 s1 = project(u1), s2 = project(u2);
  const T2 det = cross2(s1, s2);
  // the lines the facet spans; a crossing outside them starts the scan at
  // the nearest, since every line between holds no point of the facet
  T3 low{}, high{};
  bool spanned = false;
  for (const auto &triangle : facet)
    for (std::size_t k = 0; k < 3; ++k) {
      const T3 j(cross2(s1, from(origin, corner(std::size_t(triangle[k])))) /
                 det);
      low = spanned && low < j ? low : j;
      high = spanned && high > j ? high : j;
      spanned = true;
    }
  const T3 start = tau < low ? low : tau > high ? high : tau;
  const T3 along = dot3(num, u1);
  const T3 skew = T3(den) * dot(u1, u2);
  const T3 length = T3(den) * dot(u1, u1);
  const T3 step =
      (rise - tau * scale < T3(0)) == (scale < T3(0)) ? T3(1) : T3(-1);
  const auto line = [&](const T3 &j) {
    const T3 foot = along - j * skew;
    const T3 target = round_div(foot, length);
    bool found = false;
    T3 best{}, best_gap{};
    for (const auto &triangle : facet) {
      const std::array<tf::exact::pt2<Coord>, 3> q{
          corner(std::size_t(triangle[0])), corner(std::size_t(triangle[1])),
          corner(std::size_t(triangle[2]))};
      const bool flip = cross2(from(q[0], q[1]), from(q[0], q[2])) < T2(0);
      bool has_lo = false, has_hi = false, empty = false;
      T3 lo{}, hi{};
      for (std::size_t k = 0; k < 3 && !empty; ++k) {
        const auto edge = from(q[k], q[(k + 1) % 3]);
        T3 rest =
            T3(cross2(edge, from(q[k], origin))) + j * T3(cross2(edge, s2));
        T3 slope(cross2(edge, s1));
        if (flip) {
          rest = -rest;
          slope = -slope;
        }
        if (slope > T3(0)) {
          const T3 bound = -floor_div(rest, slope);
          lo = has_lo && lo > bound ? lo : bound;
          has_lo = true;
        } else if (slope < T3(0)) {
          const T3 bound = floor_div(rest, -slope);
          hi = has_hi && hi < bound ? hi : bound;
          has_hi = true;
        } else
          empty = rest < T3(0);
      }
      if (empty || !has_lo || !has_hi || lo > hi)
        continue;
      if (!inside(lo, j))
        lo = lo + T3(1);
      if (lo <= hi && !inside(hi, j))
        hi = hi - T3(1);
      if (lo > hi)
        continue;
      const T3 i = target < lo ? lo : target > hi ? hi : target;
      if (!inside(i, j))
        continue;
      T3 gap = i * length - foot;
      gap = gap < T3(0) ? -gap : gap;
      if (!found || gap < best_gap) {
        best = i;
        best_gap = gap;
        found = true;
      }
    }
    return found && state(best, j);
  };
  for (T3 k(0); start + k <= high || start - k >= low; k = k + T3(1)) {
    const T3 next = start + k * step, back = start - k * step;
    if (next >= low && next <= high && line(next))
      return true;
    if (k != T3(0) && back >= low && back <= high && line(back))
      return true;
  }
  return false;
}

} // namespace tf::topology::cdt::dt3

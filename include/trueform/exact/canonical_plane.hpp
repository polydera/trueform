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

#include "../core/point.hpp"
#include "./meta.hpp"
#include "./plane_support.hpp"
#include "./vertex.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace tf::exact {

/// A supporting plane's NAME: its normal reduced by the gcd of the three
/// components, the sign fixed so the first nonzero one is positive, and
/// the offset `N . p` carried through that same reduction.
template <typename Int>
using canonical_plane = std::array<typename meta<Int>::T2, 4>;

/// THE EQUIVALENCE: two lattice planes coincide, up to scale and
/// orientation, exactly when one integer normal is a rational multiple of
/// the other and the offsets follow that same multiple; the reduced,
/// sign-fixed quadruple is the unique representative of that ray, so key
/// equality IS coplanarity BY CONSTRUCTION and a plain lexicographic sort
/// gathers one plane's carriers into one run.
///
/// THE ONE PRODUCER of that representative. The reduction divides the
/// offset exactly whenever the offset is the normal against a lattice
/// point of the plane, and every quadruple stated here is: the namer's
/// offset is the normal against the support's own point, the quantizer's
/// is its direction against its own intercept.
///
/// The zero quadruple is a name no plane can take, so a carrier with no
/// plane keeps it unchanged.
template <typename Int>
auto canonicalize_plane(canonical_plane<Int> plane) -> canonical_plane<Int> {
  using T2 = typename meta<Int>::T2;
  using T1 = typename meta<Int>::T1;
  const auto magnitude = [](const T2 &v) { return v < T2(0) ? -v : v; };
  // the trailing zeros are counted on the low word, so a positive wide
  // value is shifted once per step of the reduction instead of once per bit
  const auto make_odd = [](auto &v) -> unsigned {
    unsigned above = 0;
    while (std::uint64_t(v) == 0) {
      v >>= 64;
      above += 64;
    }
    auto word = std::uint64_t(v);
    unsigned zeros = 0;
    while ((word & 1u) == 0) {
      word >>= 1;
      ++zeros;
    }
    v >>= zeros;
    return above + zeros;
  };
  // THE REDUCTION RUNS AT THE WIDTH ITS OPERANDS OCCUPY. The ladder is the
  // same algorithm at every rung: the loop already drops to the machine word
  // as soon as the larger operand fits in one, and the rung below the widest
  // is reached the same way, by the caller narrowing operands that fit.
  const auto gcd = [&make_odd](auto a, auto b) {
    using wide_t = decltype(a);
    if (a == wide_t(0))
      return b;
    if (b == wide_t(0))
      return a;
    const unsigned twos_a = make_odd(a), twos_b = make_odd(b);
    const unsigned twos = twos_a < twos_b ? twos_a : twos_b;
    for (;;) {
      if (b < a) {
        const wide_t lo = b;
        b = a;
        a = lo;
      }
      // the machine word is where a subtraction is one instruction, so the
      // reduction finishes there
      if (wide_t(std::uint64_t(b)) == b) {
        auto u = std::uint64_t(a), v = std::uint64_t(b);
        while (v != u) {
          if (v < u) {
            const auto lo = v;
            v = u;
            u = lo;
          }
          v -= u;
          while ((v & 1u) == 0)
            v >>= 1;
        }
        return wide_t(u) << twos;
      }
      // above the word a division carries the reduction to the word in a
      // step or two, where a subtraction would spend one per bit
      b = b % a;
      if (b == wide_t(0))
        return a << twos;
      make_odd(b);
    }
  };

  for (std::size_t k = 0; k < 3; ++k)
    if (plane[k] != T2(0)) {
      if (plane[k] < T2(0))
        for (auto &c : plane)
          c = -c;
      break;
    }
  const auto n0 = magnitude(plane[0]), n1 = magnitude(plane[1]),
             n2 = magnitude(plane[2]);
  T2 g;
  // A NORMAL IS A CROSS OF COORDINATE DIFFERENCES, so it occupies twice a
  // difference and not twice a coordinate: a mesh whose faces are small
  // against its own extent names every plane a rung below the offset's
  // width, and only the offset is ever divided wide.
  if constexpr (sizeof(T1) > sizeof(std::uint64_t)) {
    const auto narrow = [](const T2 &v) { return T2(T1(v)) == v; };
    if (narrow(n0) && narrow(n1) && narrow(n2)) {
      T1 reduced = gcd(T1(n0), T1(n1));
      if (reduced != T1(1))
        reduced = gcd(reduced, T1(n2));
      g = T2(reduced);
    } else {
      g = gcd(n0, n1);
      if (g != T2(1))
        g = gcd(g, n2);
    }
  } else {
    g = gcd(n0, n1);
    if (g != T2(1))
      g = gcd(g, n2);
  }
  if (g > T2(1))
    for (auto &c : plane)
      c = c / g;
  return plane;
}

/// A supporting plane's NAME, read off the cross the support already
/// computed and the point it stands on.
///
/// A support of fewer than three points has no plane and names the zero
/// quadruple; callers that must keep such carriers apart read the
/// support's own verdict, as the plane producers already do.
template <typename Int>
auto make_canonical_plane(const plane_support<Int> &support)
    -> canonical_plane<Int> {
  using T2 = typename meta<Int>::T2;
  const auto &p = support.point[0];
  return canonicalize_plane<Int>(
      {support.normal[0], support.normal[1], support.normal[2],
       support.normal[0] * T2(p[0]) + support.normal[1] * T2(p[1]) +
           support.normal[2] * T2(p[2])});
}

/// `N . x - D` of the named plane. A name carries its offset against the
/// lattice origin, so the value needs no base point and a caller holding
/// only the name tests any point against it.
template <typename Int>
auto orient3d_plane_value(const canonical_plane<Int> &plane, const pt3<Int> &x)
    -> typename meta<Int>::T2 {
  using T2 = typename meta<Int>::T2;
  return plane[0] * T2(x[0]) + plane[1] * T2(x[1]) + plane[2] * T2(x[2]) -
         plane[3];
}

/// Which side of the named plane `x` lies on; `0` is ON it.
template <typename Int>
auto orient3d_plane_sign(const canonical_plane<Int> &plane, const pt3<Int> &x)
    -> int {
  const auto v = orient3d_plane_value(plane, x);
  return (v > 0) ? 1 : (v < 0) ? -1 : 0;
}

} // namespace tf::exact

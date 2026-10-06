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

#include "./float_filter_sound.hpp"
#include "./insphere/filter_sign.hpp"
#include "./int32.hpp"
#include "./int64.hpp"
#include "./meta.hpp"
#include "./orient3d.hpp"
#include "./vertex.hpp"

#include <array>
#include <utility>

namespace tf::exact {

/// Magnitude bits each rung of the determinant occupies over a lattice of
/// `lattice_bits` value bits. A difference of two coordinates spans one bit
/// more than a coordinate; a lift is three of their squares; a 3x3 minor of
/// difference rows is six of their triple products; and the determinant is
/// four lifts against four minors, 72 of those products, which is what puts
/// the value seven bits above five differences.
constexpr auto insphere_difference_bits(int lattice_bits) -> int {
  return lattice_bits + 1;
}

constexpr auto insphere_lift_bits(int lattice_bits) -> int {
  return 2 * (lattice_bits + 1) + 2;
}

constexpr auto insphere_minor_bits(int lattice_bits) -> int {
  return 3 * (lattice_bits + 1) + 3;
}

constexpr auto insphere_bits(int lattice_bits) -> int {
  return 5 * (lattice_bits + 1) + 7;
}

/// Whether every rung the determinant is built on holds a lattice of
/// `lattice_bits` value bits: differences on `meta<Int>::T1`, lifts and
/// minors on `meta<Int>::T2`, the value on `meta<Int>::T3`.
template <typename Int>
constexpr auto insphere_admits(int lattice_bits) -> bool {
  return insphere_difference_bits(lattice_bits) <= meta<Int>::t1_bits &&
         insphere_lift_bits(lattice_bits) <= meta<Int>::t2_bits &&
         insphere_minor_bits(lattice_bits) <= meta<Int>::t2_bits &&
         insphere_bits(lattice_bits) <= meta<Int>::t3_bits;
}

static_assert(insphere_admits<int32>(meta<int32>::coordinate_bits),
              "the int32 lattice must fit its own insphere determinant");
static_assert(insphere_admits<int64>(meta<int64>::coordinate_bits),
              "the int64 lattice must fit its own insphere determinant");

/// Bits of uniform scale above the `Int` lattice those same rungs still
/// admit. A tier that carries scaled coordinates states its own depth cap
/// against this one instead of restating the widths.
template <typename Int> constexpr auto insphere_scale_bits() -> int {
  int scale = 0;
  while (insphere_admits<Int>(meta<Int>::coordinate_bits + scale + 1))
    ++scale;
  return scale;
}

/// Exact insphere value of five points carried in one common positive
/// multiple of the `Int` lattice: MINUS the determinant of the four rows
/// `[x - e, |x - e|^2]`, x running a, b, c, d.
///
/// The leading minus is part of the definition. With it, a tet whose
/// @ref tf::exact::orient3d_value is positive reads a positive value
/// exactly when e is strictly inside its circumsphere, a negative one when
/// e is outside, and zero when the five points are cospherical. Rows carry
/// one site each, so the value is antisymmetric in all FIVE arguments —
/// the query is a row like any other.
///
/// The value scales with the fifth power of the common multiple, so a
/// caller carrying scaled coordinates states its scale against
/// @ref tf::exact::insphere_scale_bits.
template <typename Int, typename Coord>
auto insphere_value_scaled(const pt3<Coord> &a, const pt3<Coord> &b,
                           const pt3<Coord> &c, const pt3<Coord> &d,
                           const pt3<Coord> &e) ->
    typename meta<Int>::T3 {
  using T1 = typename meta<Int>::T1;
  using T2 = typename meta<Int>::T2;
  using value_type = typename meta<Int>::T3;

  T1 ax = T1(a[0]) - e[0], ay = T1(a[1]) - e[1], az = T1(a[2]) - e[2];
  T1 bx = T1(b[0]) - e[0], by = T1(b[1]) - e[1], bz = T1(b[2]) - e[2];
  T1 cx = T1(c[0]) - e[0], cy = T1(c[1]) - e[1], cz = T1(c[2]) - e[2];
  T1 dx = T1(d[0]) - e[0], dy = T1(d[1]) - e[1], dz = T1(d[2]) - e[2];

  T2 ab = T2(ax) * by - T2(bx) * ay;
  T2 bc = T2(bx) * cy - T2(cx) * by;
  T2 cd = T2(cx) * dy - T2(dx) * cy;
  T2 da = T2(dx) * ay - T2(ax) * dy;
  T2 ac = T2(ax) * cy - T2(cx) * ay;
  T2 bd = T2(bx) * dy - T2(dx) * by;

  T2 abc = T2(az) * bc - T2(bz) * ac + T2(cz) * ab;
  T2 bcd = T2(bz) * cd - T2(cz) * bd + T2(dz) * bc;
  T2 cda = T2(cz) * da + T2(dz) * ac + T2(az) * cd;
  T2 dab = T2(dz) * ab + T2(az) * bd + T2(bz) * da;

  T2 alift = T2(ax) * ax + T2(ay) * ay + T2(az) * az;
  T2 blift = T2(bx) * bx + T2(by) * by + T2(bz) * bz;
  T2 clift = T2(cx) * cx + T2(cy) * cy + T2(cz) * cz;
  T2 dlift = T2(dx) * dx + T2(dy) * dy + T2(dz) * dz;

  return value_type(alift) * value_type(bcd) -
         value_type(blift) * value_type(cda) +
         value_type(clift) * value_type(dab) -
         value_type(dlift) * value_type(abc);
}

/// Exact insphere value on the `Int` lattice itself.
template <typename Int>
auto insphere_value(const pt3<Int> &a, const pt3<Int> &b, const pt3<Int> &c,
                    const pt3<Int> &d, const pt3<Int> &e) ->
    typename meta<Int>::T3 {
  return insphere_value_scaled<Int>(a, b, c, d, e);
}

/// Exact insphere sign (no SoS) of scaled points. Returns -1, 0, or +1.
///
/// The value lands on `meta<Int>::T3`, a software-wide integer on either
/// lattice, so @ref tf::exact::insphere::filter_sign
/// answers first wherever @ref tf::exact::float_filter_sound holds, and
/// the exact value decides what the filter leaves undecided. The sign is
/// exact on every path.
template <typename Int, typename Coord>
auto insphere_sign_scaled(const pt3<Coord> &a, const pt3<Coord> &b,
                          const pt3<Coord> &c, const pt3<Coord> &d,
                          const pt3<Coord> &e) -> int {
  if constexpr (float_filter_sound) {
    const int filtered = insphere::filter_sign<Int>(a, b, c, d, e);
    if (filtered)
      return filtered;
  }
  const auto value = insphere_value_scaled<Int>(a, b, c, d, e);
  return (value > 0) ? 1 : (value < 0) ? -1 : 0;
}

/// Exact insphere sign (no SoS). Returns -1, 0, or +1.
template <typename Int>
auto insphere_sign(const pt3<Int> &a, const pt3<Int> &b, const pt3<Int> &c,
                   const pt3<Int> &d, const pt3<Int> &e) -> int {
  return insphere_sign_scaled<Int>(a, b, c, d, e);
}

/// Insphere sign of five identified sites under lift-only Simulation of
/// Simplicity, the sites carried in one common positive multiple of the
/// `Int` lattice. Returns -1, 0, or +1 in the sign convention of
/// @ref tf::exact::insphere_value_scaled.
///
/// Coordinates are never perturbed; site i's LIFT alone carries the
/// symbolic weight `|p_i|^2 - e(i)`, `e(i) = eps^(2^k)` with k the site's
/// rank in ascending id order. Every weight lives in the ONE lift column,
/// and a determinant monomial takes a single entry per column, so the
/// expansion is exact rather than truncated and carries no mixed term:
///
///     value = insphere_value + SUM e(i) * (-1)^i * orient3d_value(the
///             other four sites, left in their caller row order)
///
/// with `i` the site's zero-based row in this call. Rows are never
/// permuted — rank selects which coefficient is read first, not where a
/// site sits. A lower rank carries the larger eps, so the first nonzero
/// term walking the sites in ascending id decides.
///
/// Total for the tier's own domain: four distinct, geometrically
/// noncoplanar tet corners and a distinct query. The query is row 4, so
/// its coefficient is exactly the tet's `orient3d_value` and the walk
/// cannot pass all five. A fully degenerate five-site input can vanish in
/// every term and reads 0; that is the caller's invariant, not this
/// predicate's.
template <typename Int, typename Index, typename Coord>
auto insphere_sos_scaled(const vertex<Index, Coord> &a,
                         const vertex<Index, Coord> &b,
                         const vertex<Index, Coord> &c,
                         const vertex<Index, Coord> &d,
                         const vertex<Index, Coord> &e) -> int {
  const int geometric = insphere_sign_scaled<Int>(a.pt, b.pt, c.pt, d.pt, e.pt);
  if (geometric)
    return geometric;

  const std::array<const vertex<Index, Coord> *, 5> vs{&a, &b, &c, &d, &e};
  std::array<int, 5> order = {0, 1, 2, 3, 4};
  for (int i = 0; i < 4; ++i)
    for (int j = i + 1; j < 5; ++j)
      if (vs[order[j]]->id < vs[order[i]]->id)
        std::swap(order[i], order[j]);

  for (int k = 0; k < 5; ++k) {
    const int row = order[k];
    std::array<pt3<Coord>, 4> rest;
    int n = 0;
    for (int i = 0; i < 5; ++i)
      if (i != row)
        rest[n++] = vs[i]->pt;
    const auto term =
        orient3d_value_scaled<Int>(rest[0], rest[1], rest[2], rest[3]);
    if (term) {
      const int sign = (term > 0) ? 1 : -1;
      return (row % 2 == 0) ? sign : -sign;
    }
  }
  return 0;
}

/// Insphere sign of five identified sites carried contiguously.
template <typename Int, typename Index, typename Coord>
auto insphere_sos_scaled(const vertex<Index, Coord> *vs) -> int {
  return insphere_sos_scaled<Int>(vs[0], vs[1], vs[2], vs[3], vs[4]);
}

/// Insphere sign of five identified sites under lift-only SoS.
template <typename Index, typename Int>
auto insphere_sos(const vertex<Index, Int> *vs) -> int {
  return insphere_sos_scaled<Int>(vs);
}

template <typename Index, typename Int>
auto insphere_sos(const std::array<vertex<Index, Int>, 5> &vs) -> int {
  return insphere_sos(vs.data());
}

/// Whether the fifth site conflicts with the circumsphere of the first
/// four, normalized by the `orientation` sign the caller stores for that
/// tet. The tet's orientation is the caller's fact, so the predicate reads
/// it rather than deriving it a second time.
template <typename Int, typename Index, typename Coord>
auto insphere_conflict_scaled(
    const vertex<Index, Coord> &a, const vertex<Index, Coord> &b,
    const vertex<Index, Coord> &c, const vertex<Index, Coord> &d,
    const vertex<Index, Coord> &e, int orientation) -> bool {
  return insphere_sos_scaled<Int>(a, b, c, d, e) * orientation > 0;
}

template <typename Int, typename Index, typename Coord>
auto insphere_conflict_scaled(const vertex<Index, Coord> *vs, int orientation)
    -> bool {
  return insphere_conflict_scaled<Int>(vs[0], vs[1], vs[2], vs[3], vs[4],
                                       orientation);
}

/// Whether the fifth site conflicts with the circumsphere of the first four.
template <typename Index, typename Int>
auto insphere_conflict(const vertex<Index, Int> *vs, int orientation) -> bool {
  return insphere_conflict_scaled<Int>(vs, orientation);
}

template <typename Index, typename Int>
auto insphere_conflict(const std::array<vertex<Index, Int>, 5> &vs,
                       int orientation) -> bool {
  return insphere_conflict(vs.data(), orientation);
}

} // namespace tf::exact

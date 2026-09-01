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

#include "../../exact/det2_sign.hpp"
#include "../../exact/edge_parameter.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/vertex.hpp"
#include "./plane_face_support.hpp"
#include "./plane_point_generator.hpp"
#include <array>
#include <cstddef>

namespace tf::intersect::graph {

/// The carrier line of the two faces that produce an intersection edge,
/// carried as `d = n_f x n_g = a * b1 - b * b2` with `b1`, `b2` the
/// partner's edge vectors and `a = det(a1, a2, b2)`,
/// `b = det(a1, a2, b1)`.
///
/// The identity is the width discipline: `d` itself is degree four and
/// does NOT fit the T2 rung, while every operand of this form does, so
/// each sign below is one @ref tf::exact::det2_sign away.
template <typename Int> struct plane_pair_carrier {
  std::array<typename tf::exact::meta<Int>::T1, 3> b1{}, b2{};
  typename tf::exact::meta<Int>::T2 a{}, b{};
  bool valid = false;
};

template <typename Int>
auto plane_carrier_cross(
    const std::array<typename tf::exact::meta<Int>::T1, 3> &x,
    const std::array<typename tf::exact::meta<Int>::T1, 3> &y)
    -> std::array<typename tf::exact::meta<Int>::T2, 3> {
  using T2 = typename tf::exact::meta<Int>::T2;
  return {T2(x[1]) * T2(y[2]) - T2(x[2]) * T2(y[1]),
          T2(x[2]) * T2(y[0]) - T2(x[0]) * T2(y[2]),
          T2(x[0]) * T2(y[1]) - T2(x[1]) * T2(y[0])};
}

template <typename Int, typename Vector0, typename Vector1>
auto plane_carrier_dot(const Vector0 &x, const Vector1 &y) ->
    typename tf::exact::meta<Int>::T2 {
  using T2 = typename tf::exact::meta<Int>::T2;
  return T2(x[0]) * T2(y[0]) + T2(x[1]) * T2(y[1]) + T2(x[2]) * T2(y[2]);
}

/// sign(dot(d, x)) for any operand of degree two or less.
template <typename Int, typename Vector>
auto plane_carrier_dot_sign(const plane_pair_carrier<Int> &carrier,
                            const Vector &x) -> int {
  return tf::exact::det2_sign<Int>(
      carrier.a, plane_carrier_dot<Int>(x, carrier.b1), carrier.b,
      plane_carrier_dot<Int>(x, carrier.b2));
}

/// The two edge vectors whose cross is a face's exact normal, read off
/// the points its carrier stands on — so the normal is the ORIGINAL
/// winding's.
template <typename Int, typename Face, typename GetVertex>
auto plane_face_basis(const Face &face, const GetVertex &get_vertex,
                      std::array<typename tf::exact::meta<Int>::T1, 3> &e0,
                      std::array<typename tf::exact::meta<Int>::T1, 3> &e1)
    -> bool {
  using T1 = typename tf::exact::meta<Int>::T1;
  tf::exact::pt3<Int> a{}, b{}, c{};
  if (!plane_face_support<Int>(face, get_vertex, a, b, c))
    return false;
  e0 = {T1(b[0]) - a[0], T1(b[1]) - a[1], T1(b[2]) - a[2]};
  e1 = {T1(c[0]) - a[0], T1(c[1]) - a[1], T1(c[2]) - a[2]};
  return true;
}

template <typename Int>
auto make_plane_pair_carrier(
    const std::array<typename tf::exact::meta<Int>::T1, 3> &a1,
    const std::array<typename tf::exact::meta<Int>::T1, 3> &a2,
    const std::array<typename tf::exact::meta<Int>::T1, 3> &b1,
    const std::array<typename tf::exact::meta<Int>::T1, 3> &b2)
    -> plane_pair_carrier<Int> {
  plane_pair_carrier<Int> carrier;
  const auto normal = plane_carrier_cross<Int>(a1, a2);
  carrier.b1 = b1;
  carrier.b2 = b2;
  carrier.a = plane_carrier_dot<Int>(normal, b2);
  carrier.b = plane_carrier_dot<Int>(normal, b1);
  carrier.valid = true;
  return carrier;
}

/// Sign of the key-order direction of an intersection edge against its
/// producing pair's carrier line: +1 along +(n_f x n_g), -1 against, 0
/// when the generators do not decide it.
///
/// Both endpoints lie on that line, so the question is their ORDER on
/// it, and a parameterized endpoint answers it without ever being
/// materialized: the plane through its own carrier with normal
/// `w x e_k` annihilates its parameter exactly, leaving the other
/// endpoint's position against that plane — one determinant sign — as
/// the whole decision. Where no carrier is transversal to the line the
/// endpoints are collinear with it and the carrier's own parameter
/// order answers instead.
template <typename Index, typename Int>
auto plane_edge_carrier_sign(const plane_pair_carrier<Int> &carrier,
                             const plane_point_generator<Index, Int> &lo,
                             const plane_point_generator<Index, Int> &hi)
    -> int {
  using T1 = typename tf::exact::meta<Int>::T1;
  using T2 = typename tf::exact::meta<Int>::T2;
  if (!carrier.valid)
    return 0;
  auto diff = [](const tf::exact::pt3<Int> &p, const tf::exact::pt3<Int> &q) {
    return std::array<T1, 3>{T1(p[0]) - q[0], T1(p[1]) - q[1], T1(p[2]) - q[2]};
  };
  if (!lo.on_edge && !hi.on_edge)
    return plane_carrier_dot_sign<Int>(carrier, diff(hi.u, lo.u));

  for (int side = 0; side < 2; ++side) {
    const auto &reference = side == 0 ? lo : hi;
    const auto &other = side == 0 ? hi : lo;
    const int orientation = side == 0 ? 1 : -1;
    if (!reference.on_edge)
      continue;
    const auto w = diff(reference.v, reference.u);
    const auto b1w = plane_carrier_cross<Int>(carrier.b1, w);
    const auto b2w = plane_carrier_cross<Int>(carrier.b2, w);
    for (int k = 0; k < 3; ++k) {
      // (d x w)[k] = dot(w x e_k, d): zero exactly when the carrier is
      // parallel to the line in this axis' plane
      const int transversal =
          tf::exact::det2_sign<Int>(carrier.a, b1w[k], carrier.b, b2w[k]);
      if (transversal == 0)
        continue;
      std::array<T1, 3> m{T1(0), T1(0), T1(0)};
      m[std::size_t((k + 1) % 3)] = w[std::size_t((k + 2) % 3)];
      m[std::size_t((k + 2) % 3)] = -w[std::size_t((k + 1) % 3)];
      int position = 0;
      if (!other.on_edge) {
        const auto value =
            plane_carrier_dot<Int>(m, diff(other.u, reference.u));
        position = value > T2(0) ? 1 : (value < T2(0) ? -1 : 0);
      } else {
        position = tf::exact::det2_sign<Int>(
            other.t.den, plane_carrier_dot<Int>(m, diff(other.u, reference.u)),
            -other.t.num, plane_carrier_dot<Int>(m, diff(other.v, other.u)));
      }
      return orientation * position * transversal;
    }
  }

  // every carrier in hand is collinear with the line: order along it
  for (int side = 0; side < 2; ++side) {
    const auto &reference = side == 0 ? lo : hi;
    const auto &other = side == 0 ? hi : lo;
    const int orientation = side == 0 ? 1 : -1;
    if (!reference.on_edge)
      continue;
    const auto w = diff(reference.v, reference.u);
    const int along = plane_carrier_dot_sign<Int>(carrier, w);
    if (along == 0)
      continue;
    if (!other.on_edge)
      return orientation * along *
             tf::exact::det2_sign<Int>(
                 reference.t.den,
                 plane_carrier_dot<Int>(w, diff(other.u, reference.u)),
                 reference.t.num, plane_carrier_dot<Int>(w, w));
    if (other.carrier_u == reference.carrier_u &&
        other.carrier_v == reference.carrier_v)
      return orientation * along *
             tf::exact::compare_parameter<Int>(other.t, reference.t);
  }
  return 0;
}

} // namespace tf::intersect::graph

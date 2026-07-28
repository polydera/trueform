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

#include "../../exact/coplanar_edge_edge_point.hpp"
#include "../../exact/distance_bounds.hpp"
#include "../../exact/edge_edge_closest_point.hpp"
#include "../../exact/meta.hpp"
#include "../../exact/orient2d.hpp"
#include "../../exact/orient3d.hpp"
#include "../../exact/vertex.hpp"
#include "../../exact/vv_within.hpp"
#include <array>

namespace tf::exact {

/// Predicate kernel for intersection-detection classifiers.
///
/// Bundles a distance tolerance in int coordinate units and exposes the
/// predicate vocabulary the classifiers consume:
///
/// - `tolerance_int == 0`: bounds are 0, sign reduces to standard sign,
///   `vv_equal` is bitwise.
/// - `tolerance_int > 0`: bounds via `tf::exact::*_distance_bound`, sign
///   uses `|v| <= bound`, `vv_equal` via `tf::exact::vv_within` against
///   the squared tolerance.
///
/// Convert a world distance to int units via
/// `pt_converter::convert_tolerance` before constructing.
template <typename Int> class predicate_kernel {
public:
  predicate_kernel() = default;

  /// One int unit is added to the stored tolerance when non-zero — grace
  /// against the ±1 ulp float→int converter rounding, so a value at the
  /// user's stated tolerance lands strictly inside the band.
  ///
  /// The squared tolerance is held in T1 (int64 for int32 input, int128
  /// for int64 input): `Int^2` always fits in T1, and `vv_within`
  /// promotes to T2 for the squared-distance comparison.
  explicit predicate_kernel(Int tolerance_int)
      : _tolerance_int(tolerance_int + Int(tolerance_int != Int(0))),
        _tolerance_double(static_cast<double>(_tolerance_int)),
        _tolerance_int2(typename tf::exact::meta<Int>::T1(_tolerance_int) *
                        typename tf::exact::meta<Int>::T1(_tolerance_int)),
        _is_tolerated(_tolerance_int > Int(0)) {}

  auto is_tolerated() const -> bool { return _is_tolerated; }

  auto tolerance_int() const -> Int { return _tolerance_int; }

  auto tolerance_double() const -> double { return _tolerance_double; }

  auto tolerance_int2() const -> typename tf::exact::meta<Int>::T1 {
    return _tolerance_int2;
  }

  /// Reusable bound for `orient3d(a, b, c, *)` over many `d`.
  auto orient3d_bound(const tf::exact::pt3<Int> &a,
                      const tf::exact::pt3<Int> &b,
                      const tf::exact::pt3<Int> &c) const ->
      typename tf::exact::meta<Int>::T2 {
    if (!_is_tolerated)
      return typename tf::exact::meta<Int>::T2(0);
    return tf::exact::orient3d_distance_bound(a, b, c, _tolerance_double);
  }

  /// Reusable bound for a precomputed plane normal over many `d`:
  /// `|N . (d - base)| <= bound` iff `d` is within the band of the plane.
  auto plane_bound(
      const std::array<typename tf::exact::meta<Int>::T2, 3> &normal) const ->
      typename tf::exact::meta<Int>::T2 {
    if (!_is_tolerated)
      return typename tf::exact::meta<Int>::T2(0);
    return tf::exact::orient3d_plane_distance_bound<Int>(normal,
                                                         _tolerance_double);
  }

  /// Reusable bound for `orient2d(a, b, *)` over many `c`.
  auto orient2d_bound(const tf::exact::pt2<Int> &a,
                      const tf::exact::pt2<Int> &b) const ->
      typename tf::exact::meta<Int>::T2 {
    if (!_is_tolerated)
      return typename tf::exact::meta<Int>::T2(0);
    return tf::exact::orient2d_distance_bound(a, b, _tolerance_double);
  }

  /// 3D-edge projected to axes (ax0, ax1).
  auto orient2d_bound(const tf::exact::pt3<Int> &a,
                      const tf::exact::pt3<Int> &b, int ax0, int ax1) const ->
      typename tf::exact::meta<Int>::T2 {
    if (!_is_tolerated)
      return typename tf::exact::meta<Int>::T2(0);
    return tf::exact::orient2d_distance_bound(a, b, ax0, ax1,
                                              _tolerance_double);
  }

  /// Skew-line bound: use when the orient3d call tests "are these 4
  /// points coplanar", not "is `d` on the plane of (a, b, c)".
  auto orient3d_skew_bound(const tf::exact::pt3<Int> &a,
                           const tf::exact::pt3<Int> &b,
                           const tf::exact::pt3<Int> &c,
                           const tf::exact::pt3<Int> &d) const ->
      typename tf::exact::meta<Int>::T2 {
    if (!_is_tolerated)
      return typename tf::exact::meta<Int>::T2(0);
    return tf::exact::ee_distance_bound(a, b, c, d, _tolerance_double);
  }

  auto sign(typename tf::exact::meta<Int>::T2 value,
            typename tf::exact::meta<Int>::T2 bound) const -> int {
    return (value > bound) - (value < -bound);
  }

  /// One-off sign: implicit per-call bound (not amortized).
  auto orient3d_sign(const tf::exact::pt3<Int> &a,
                     const tf::exact::pt3<Int> &b,
                     const tf::exact::pt3<Int> &c,
                     const tf::exact::pt3<Int> &d) const -> int {
    return sign(tf::exact::orient3d_value(a, b, c, d), orient3d_bound(a, b, c));
  }

  auto orient2d_sign(const tf::exact::pt2<Int> &a,
                     const tf::exact::pt2<Int> &b,
                     const tf::exact::pt2<Int> &c) const -> int {
    return sign(tf::exact::orient2d(a, b, c), orient2d_bound(a, b));
  }

  /// `vertex<Index, Int>` projected to axes (ax0, ax1).
  template <typename Index>
  auto orient2d_sign(const tf::exact::vertex<Index, Int> &v0,
                     const tf::exact::vertex<Index, Int> &v1,
                     const tf::exact::vertex<Index, Int> &v2, int ax0,
                     int ax1) const -> int {
    using T1 = typename tf::exact::meta<Int>::T1;
    using T2 = typename tf::exact::meta<Int>::T2;
    T1 ax = T1(v1.pt[ax0]) - T1(v0.pt[ax0]);
    T1 ay = T1(v1.pt[ax1]) - T1(v0.pt[ax1]);
    T1 bx = T1(v2.pt[ax0]) - T1(v0.pt[ax0]);
    T1 by = T1(v2.pt[ax1]) - T1(v0.pt[ax1]);
    T2 det = T2(ax) * T2(by) - T2(ay) * T2(bx);
    return sign(det, orient2d_bound(v0.pt, v1.pt, ax0, ax1));
  }

  auto vv_equal(const tf::exact::pt3<Int> &a,
                const tf::exact::pt3<Int> &b) const -> bool {
    if (!_is_tolerated)
      return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    return tf::exact::vv_within(
        a, b, typename tf::exact::meta<Int>::T2(_tolerance_int2));
  }

  auto vv_equal(const tf::exact::pt2<Int> &a,
                const tf::exact::pt2<Int> &b) const -> bool {
    if (!_is_tolerated)
      return a[0] == b[0] && a[1] == b[1];
    return tf::exact::vv_within(
        a, b, typename tf::exact::meta<Int>::T2(_tolerance_int2));
  }

  /// Contact point of two edges. Exact: the projected-crossing
  /// interpolation. Tolerated: the closest-approach point — a
  /// band-declared "coplanar" pair may be skew, where the projected
  /// crossing sits at an arbitrary parameter of the near-parallel
  /// corridor (gap up to tol/sin); the closest approach is within the
  /// band by the predicate that fired, and coincides with the crossing
  /// when the segments truly cross.
  template <typename Index>
  auto edge_edge_point(const tf::exact::vertex<Index, Int> &a0,
                       const tf::exact::vertex<Index, Int> &a1,
                       const tf::exact::vertex<Index, Int> &b0,
                       const tf::exact::vertex<Index, Int> &b1, int ax0,
                       int ax1) const -> tf::exact::pt3<Int> {
    if (!_is_tolerated)
      return tf::exact::coplanar_edge_edge_point(a0, a1, b0, b1, ax0, ax1);
    return tf::exact::edge_edge_closest_point(a0.pt, a1.pt, b0.pt, b1.pt);
  }

  /// Transversal flavor: in the exact regime the correct point is the
  /// already-computed plane crossing (exact zero ⇒ it lies on the face
  /// edge's line); the tolerated regime replaces it with the
  /// closest-approach point of the two edges.
  template <typename Index>
  auto edge_edge_point(const tf::exact::vertex<Index, Int> &a0,
                       const tf::exact::vertex<Index, Int> &a1,
                       const tf::exact::vertex<Index, Int> &b0,
                       const tf::exact::vertex<Index, Int> &b1,
                       const tf::exact::pt3<Int> &plane_crossing) const
      -> tf::exact::pt3<Int> {
    if (!_is_tolerated)
      return plane_crossing;
    return tf::exact::edge_edge_closest_point(a0.pt, a1.pt, b0.pt, b1.pt);
  }

  /// Do two edges actually come within the tolerance of each other?
  /// The band predicates over-accept for near-parallel pairs (bound
  /// scales with 1/sin); this is the well-conditioned confirmation.
  /// Exact regime: trust the exact zero that got the caller here.
  template <typename Index>
  auto edge_contact(const tf::exact::vertex<Index, Int> &a0,
                    const tf::exact::vertex<Index, Int> &a1,
                    const tf::exact::vertex<Index, Int> &b0,
                    const tf::exact::vertex<Index, Int> &b1) const -> bool {
    if (!_is_tolerated)
      return true;
    auto p = tf::exact::edge_edge_closest_point(a0.pt, a1.pt, b0.pt, b1.pt);
    auto q = tf::exact::segment_closest_point(b0.pt, b1.pt, p);
    return tf::exact::vv_within(
        p, q, typename tf::exact::meta<Int>::T2(_tolerance_int2));
  }

private:
  Int _tolerance_int = Int(0);
  double _tolerance_double = 0.0;
  typename tf::exact::meta<Int>::T1 _tolerance_int2 =
      typename tf::exact::meta<Int>::T1(0);
  bool _is_tolerated = false;
};

} // namespace tf::exact

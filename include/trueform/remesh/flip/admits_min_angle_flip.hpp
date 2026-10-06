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

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>

#include "../../core/angle.hpp"
#include "../../core/coordinate_type.hpp"
#include "../../core/cross.hpp"
#include "../../core/dot.hpp"
#include "../../core/none.hpp"
#include "../../core/point.hpp"
#include "../../core/points.hpp"
#include "../../core/sqrt.hpp"
#include "../../core/unsafe.hpp"
#include "../../topology/half_edge_handle.hpp"
#include "../../topology/half_edges.hpp"
#include "../feature_mask.hpp"
#include "../optimize_valence.hpp" // flip_keeps_orientation

namespace tf::remesh {

/// Minimum interior angle (radians) of triangle (A, B, C).
template <typename Real>
inline auto tri_min_angle(const tf::point<Real, 3> &A,
                          const tf::point<Real, 3> &B,
                          const tf::point<Real, 3> &C) -> Real {
  auto len = [](const tf::point<Real, 3> &p, const tf::point<Real, 3> &q) {
    Real dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
    return Real(std::sqrt(double(dx * dx + dy * dy + dz * dz)));
  };
  Real a = len(B, C), b = len(A, C), c = len(A, B);
  auto angle = [](Real opp, Real s1, Real s2) -> Real {
    if (s1 <= 0 || s2 <= 0)
      return Real(0);
    Real co = (s1 * s1 + s2 * s2 - opp * opp) / (2 * s1 * s2);
    co = std::min(Real(1), std::max(Real(-1), co));
    return Real(std::acos(double(co)));
  };
  return std::min({angle(a, b, c), angle(b, a, c), angle(c, a, b)});
}

/// Whether flipping one edge is admitted: the flip is topologically legal, it
/// raises the smaller of the two triangles' minimum angles past the
/// hysteresis margin, it displaces the surface by at most `max_deviation`
/// when one is asked for, it does not fold either triangle, and — under a
/// feature mask — a quad touching the locked net clears `angle_floor`.
///
/// This is the one producer of the verdict @ref tf::remesh::flip_min_angle
/// sweeps with.
template <typename Index, typename PointsPolicy, typename MaskOrNone>
auto admits_min_angle_flip(
    const tf::half_edges<Index> &he, const tf::points<PointsPolicy> &points,
    const MaskOrNone &mask, const tf::edge_handle<Index> &eh,
    tf::rad<tf::coordinate_type<PointsPolicy>> angle_floor,
    tf::coordinate_type<PointsPolicy> max_deviation) -> bool {
  using Real = tf::coordinate_type<PointsPolicy>;
  constexpr bool HasMask = !std::is_same_v<MaskOrNone, tf::none_t>;
  // Hysteresis: require a small gain so a flip and its reverse don't oscillate.
  const Real margin = Real(0.0174533); // 1 degree

  if (!he.is_flip_ok(eh))
    return false;
  if constexpr (HasMask)
    if (mask.is_feature(eh.id()))
      return false;
  auto a0 = he.half_edge_handle(tf::unsafe, eh, false);
  auto a1 = he.next(tf::unsafe, a0);
  auto a2 = he.next(tf::unsafe, a1);
  auto b0 = he.half_edge_handle(tf::unsafe, eh, true);
  auto b1 = he.next(tf::unsafe, b0);
  auto b2 = he.next(tf::unsafe, b1);

  auto va0 = he.start_vertex_handle(tf::unsafe, a0).id();
  auto va1 = he.start_vertex_handle(tf::unsafe, a1).id();
  auto va2 = he.start_vertex_handle(tf::unsafe, a2).id();
  auto vb2 = he.start_vertex_handle(tf::unsafe, b2).id();

  auto P = [&points](Index v) {
    auto p = points[v];
    return tf::point<double, 3>{double(p[0]), double(p[1]), double(p[2])};
  };
  auto pa0 = P(va0), pa1 = P(va1), pa2 = P(va2), pb2 = P(vb2);

  // Old triangles (va0,va1,va2) and (va1,va0,vb2); after flip (va2,vb2,va1)
  // and (vb2,va2,va0).
  double min_before =
      std::min(tri_min_angle(pa0, pa1, pa2), tri_min_angle(pa1, pa0, pb2));
  double min_after =
      std::min(tri_min_angle(pa2, pb2, pa1), tri_min_angle(pb2, pa2, pa0));

  if (min_after <= min_before + double(margin))
    return false;
  if (max_deviation > 0) {
    auto d0 = pa1 - pa0;
    auto d1 = pb2 - pa2;
    auto n = tf::cross(d0, d1);
    double n2 = tf::dot(n, n);
    double gap = n2 < 1e-30 ? std::numeric_limits<double>::max()
                            : std::abs(tf::dot(pa2 - pa0, n)) / tf::sqrt(n2);
    if (gap > double(max_deviation))
      return false;
  }
  if (!flip_keeps_orientation(points, va0, va1, va2, vb2))
    return false; // would fold the surface

  if constexpr (HasMask) {
    bool all_regular = mask.vertex_type(va0) == vertex_feature_type::regular &&
                       mask.vertex_type(va1) == vertex_feature_type::regular &&
                       mask.vertex_type(va2) == vertex_feature_type::regular &&
                       mask.vertex_type(vb2) == vertex_feature_type::regular;
    // A feature-touching flip must yield a genuinely good triangle.
    if (!all_regular && min_after < angle_floor.value)
      return false;
  }
  return true;
}

} // namespace tf::remesh

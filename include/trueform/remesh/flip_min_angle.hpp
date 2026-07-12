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

#include <cmath>
#include <limits>
#include <type_traits>

#include "../core/coordinate_type.hpp"
#include "../core/none.hpp"
#include "../core/point.hpp"
#include "../core/points.hpp"
#include "../core/angle.hpp"
#include "../topology/half_edges.hpp"
#include "./feature_mask.hpp"
#include "./optimize_valence.hpp" // detail::flip_keeps_orientation

namespace tf::remesh {

namespace detail {

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

} // namespace detail

/// @ingroup remesh
/// @brief Delaunay-style edge flips that raise the minimum angle.
///
/// Flips a (non-feature, flippable) edge when doing so raises the smaller of the
/// two triangles' minimum angles -- the criterion that removes slivers/caps that
/// a valence flip leaves behind. Every flip is fold-guarded (it must not invert
/// a triangle), so it is safe on uneven or near-flat regions where is_flip_ok's
/// topology-only check is not enough. A flip whose quad touches a feature/crease
/// vertex is only taken when the result clears `angle_floor`, so it never trades
/// one sliver for a marginally-less-bad one against the locked feature net.
///
/// @param mask  feature_mask (skip feature edges/floor near feature vertices) or
///              tf::none (no features).
/// @param max_deviation  When > 0, reject flips that displace the surface by
///              more than this: the displacement of a flip is the skew
///              distance between the old and new diagonal.
template <typename Index, typename PointsPolicy, typename MaskOrNone>
auto flip_min_angle(tf::half_edges<Index> &he,
                    const tf::points<PointsPolicy> &points,
                    const MaskOrNone &mask, int iterations,
                    tf::rad<tf::coordinate_type<PointsPolicy>> angle_floor,
                    tf::coordinate_type<PointsPolicy> max_deviation = 0)
    -> Index {
  using Real = tf::coordinate_type<PointsPolicy>;
  constexpr bool HasMask = !std::is_same_v<MaskOrNone, tf::none_t>;
  // Hysteresis: require a small gain so a flip and its reverse don't oscillate.
  const Real margin = Real(0.0174533); // 1 degree

  Index n_flipped = 0;
  for (int it = 0; it < iterations; ++it) {
    Index flipped = 0;
    for (auto eh : he.edge_handles()) {
      if (!he.is_flip_ok(eh))
        continue;
      if constexpr (HasMask)
        if (mask.is_feature(eh.id()))
          continue;
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

      auto P = [&](Index v) {
        auto p = points[v];
        return tf::point<double, 3>{double(p[0]), double(p[1]), double(p[2])};
      };
      auto pa0 = P(va0), pa1 = P(va1), pa2 = P(va2), pb2 = P(vb2);

      // Old triangles (va0,va1,va2) and (va1,va0,vb2); after flip (va2,vb2,va1)
      // and (vb2,va2,va0).
      double min_before = std::min(detail::tri_min_angle(pa0, pa1, pa2),
                                   detail::tri_min_angle(pa1, pa0, pb2));
      double min_after = std::min(detail::tri_min_angle(pa2, pb2, pa1),
                                  detail::tri_min_angle(pb2, pa2, pa0));

      if (min_after <= min_before + double(margin))
        continue;
      if (max_deviation > 0) {
        auto d0 = pa1 - pa0;
        auto d1 = pb2 - pa2;
        auto n = tf::cross(d0, d1);
        double n2 = tf::dot(n, n);
        double gap = n2 < 1e-30
                         ? std::numeric_limits<double>::max()
                         : std::abs(tf::dot(pa2 - pa0, n)) / tf::sqrt(n2);
        if (gap > double(max_deviation))
          continue;
      }
      if (!detail::flip_keeps_orientation(points, va0, va1, va2, vb2))
        continue; // would fold the surface

      if constexpr (HasMask) {
        bool all_regular =
            mask.vertex_type(va0) == vertex_feature_type::regular &&
            mask.vertex_type(va1) == vertex_feature_type::regular &&
            mask.vertex_type(va2) == vertex_feature_type::regular &&
            mask.vertex_type(vb2) == vertex_feature_type::regular;
        if (!all_regular && min_after < angle_floor.value)
          continue; // feature-touching flip must yield a genuinely good triangle
      }

      he.flip(eh);
      ++flipped;
    }
    n_flipped += flipped;
    if (flipped == 0)
      break;
  }
  return n_flipped;
}

} // namespace tf::remesh

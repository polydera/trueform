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
#include "../../core/buffer.hpp"
#include "../../core/point.hpp"
#include "../../core/polygons.hpp"
#include "../../core/range.hpp"
#include "../../exact/meta.hpp"
#include "../../topology/face_membership_like.hpp"
#include "./hole_chord_is_forbidden.hpp"
#include "./hole_lattice.hpp"
#include "./hole_metric.hpp"
#include "./hole_rim.hpp"
#include "./hole_triangles_overlap.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace tf::fill {

/// The scratch one tiny rim's enumeration walks on.
template <typename Index, typename Int> struct hole_tiny_scratch {
  tf::buffer<tf::point<Int, 3>> lattice;
  tf::buffer<std::array<Index, 3>> candidate;
  tf::buffer<std::array<Index, 3>> best;
};

/// Fill a rim of at most five positions by enumerating its triangulations.
///
/// A candidate is ADMITTED on geometry alone — every triangle stands on a
/// plane, no chord the guards forbid, no pair of triangles meeting beyond
/// the simplex they share — and only then ranked, by the default objective:
/// the largest angle the patch states first, its area second. The metric
/// never decides validity.
///
/// Returns false when no candidate is admissible, which hands the rim to the
/// next tier.
template <typename Policy, typename MembershipPolicy, typename Index,
          typename Int, typename RealT>
auto fill_tiny_hole(const tf::polygons<Policy> &polygons,
                    const tf::face_membership_like<MembershipPolicy> &fm,
                    const tf::fill::hole_rim<Index, Int, RealT> &rim,
                    tf::fill::hole_tiny_scratch<Index, Int> &scratch,
                    tf::buffer<std::array<Index, 3>> &triangles) -> bool {
  using T2 = typename tf::exact::meta<Int>::T2;
  const Index n = Index(rim.size());
  const auto next = [n](Index k) { return Index(k + 1 == n ? 0 : k + 1); };

  scratch.lattice.allocate(std::size_t(n));
  for (Index k = 0; k < n; ++k)
    scratch.lattice[std::size_t(k)] =
        tf::fill::hole_unscaled_point<Int>(rim.sites[std::size_t(k)].pt);
  const auto lattice = tf::make_range(scratch.lattice);

  const auto position = [&rim](Index k) {
    return rim.positions[std::size_t(k)].template as<double>();
  };
  const auto normal_of = [&rim, &position](const std::array<Index, 3> &t) {
    const int first = tf::fill::hole_canonical_rotation(
        rim.corners[std::size_t(t[0])], rim.corners[std::size_t(t[1])],
        rim.corners[std::size_t(t[2])]);
    return tf::fill::hole_oriented_normal(
        position(t[std::size_t(first)]),
        position(t[std::size_t((first + 1) % 3)]),
        position(t[std::size_t((first + 2) % 3)]));
  };

  bool any = false;
  double best_angle = 0.0;
  double best_area = 0.0;
  const Index roots = Index(n == 3 ? 1 : n == 4 ? 2 : n);
  for (Index root = 0; root < roots; ++root) {
    scratch.candidate.clear();
    for (Index step = 1; step + 1 < n; ++step) {
      const Index x = Index((root + step) % n);
      const Index y = Index((root + step + 1) % n);
      scratch.candidate.push_back({y, x, root});
    }

    bool admissible = true;
    for (auto &triangle : scratch.candidate) {
      const auto &a = lattice[std::size_t(triangle[0])];
      const auto &b = lattice[std::size_t(triangle[1])];
      const auto &c = lattice[std::size_t(triangle[2])];
      using T1 = typename tf::exact::meta<Int>::T1;
      const T1 ux = T1(b[0]) - a[0], uy = T1(b[1]) - a[1], uz = T1(b[2]) - a[2];
      const T1 vx = T1(c[0]) - a[0], vy = T1(c[1]) - a[1], vz = T1(c[2]) - a[2];
      if (T2(uy) * vz - T2(uz) * vy == T2(0) &&
          T2(uz) * vx - T2(ux) * vz == T2(0) &&
          T2(ux) * vy - T2(uy) * vx == T2(0))
        admissible = false;
      if (!tf::fill::hole_normal_is_usable(normal_of(triangle)))
        admissible = false;
    }
    for (Index step = 2; admissible && step + 1 < n; ++step)
      if (tf::fill::hole_chord_is_forbidden(polygons, fm, rim, root,
                                            Index((root + step) % n)))
        admissible = false;
    for (std::size_t i = 0; admissible && i < scratch.candidate.size(); ++i)
      for (std::size_t j = i + 1; admissible && j < scratch.candidate.size();
           ++j)
        if (tf::fill::hole_triangles_overlap(lattice, scratch.candidate[i],
                                             scratch.candidate[j]))
          admissible = false;
    if (!admissible)
      continue;

    double angle = 0.0;
    double area = 0.0;
    for (std::size_t i = 0; i < scratch.candidate.size(); ++i) {
      const auto &triangle = scratch.candidate[i];
      const auto normal = normal_of(triangle);
      area += tf::fill::hole_triangle_area(normal);
      for (int e = 0; e < 3; ++e) {
        const Index x = triangle[std::size_t(e)];
        const Index y = triangle[std::size_t((e + 1) % 3)];
        if (next(y) == x) {
          angle = std::max(angle,
                           tf::fill::hole_dihedral_angle(
                               normal, rim.edges[std::size_t(y)].normal));
          continue;
        }
        for (std::size_t j = i + 1; j < scratch.candidate.size(); ++j) {
          const auto &other = scratch.candidate[j];
          for (int f = 0; f < 3; ++f)
            if (other[std::size_t(f)] == y &&
                other[std::size_t((f + 1) % 3)] == x)
              angle = std::max(angle, tf::fill::hole_dihedral_angle(
                                          normal, normal_of(other)));
        }
      }
    }
    if (!std::isfinite(angle) || !std::isfinite(area))
      continue;
    if (!any || angle < best_angle ||
        (angle == best_angle && area < best_area)) {
      any = true;
      best_angle = angle;
      best_area = area;
      scratch.best.clear();
      for (const auto &triangle : scratch.candidate)
        scratch.best.push_back(triangle);
    }
  }

  if (!any)
    return false;
  for (const auto &triangle : scratch.best)
    triangles.push_back({rim.corners[std::size_t(triangle[0])],
                         rim.corners[std::size_t(triangle[1])],
                         rim.corners[std::size_t(triangle[2])]});
  return true;
}

} // namespace tf::fill

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
#include "../../core/cross.hpp"
#include "../../core/dot.hpp"
#include "../../core/point.hpp"
#include "../../core/vector.hpp"
#include <cmath>

namespace tf::fill {

/// Which of three triangle slots carries the smallest key, which is the
/// rotation every numerical reading of that triangle is taken in — winding
/// preserved, so one triangle reads one value however it is named.
template <typename Index>
auto hole_canonical_rotation(Index k0, Index k1, Index k2) -> int {
  if (k0 <= k1 && k0 <= k2)
    return 0;
  return k1 <= k2 ? 1 : 2;
}

/// The doubled-area normal of the oriented triangle `a b c`.
inline auto hole_oriented_normal(const tf::point<double, 3> &a,
                                 const tf::point<double, 3> &b,
                                 const tf::point<double, 3> &c)
    -> tf::vector<double, 3> {
  return tf::cross(b - a, c - a);
}

/// Whether a triangle states a plane its metric can be read from. A normal
/// that is not finite, or has no length at all, states none — which makes
/// the triangle inadmissible exactly as a guard failure does.
inline auto hole_normal_is_usable(const tf::vector<double, 3> &normal)
    -> bool {
  const double length2 = tf::dot(normal, normal);
  return std::isfinite(length2) && length2 > 0.0;
}

/// A triangle's area, from its doubled-area normal.
inline auto hole_triangle_area(const tf::vector<double, 3> &normal) -> double {
  return 0.5 * std::sqrt(tf::dot(normal, normal));
}

/// The angle between two triangles wound consistently across their shared
/// edge, in `[0, pi]`: zero where they lie flat and pi where they fold back
/// onto one another.
inline auto hole_dihedral_angle(const tf::vector<double, 3> &n0,
                                const tf::vector<double, 3> &n1) -> double {
  const auto axis = tf::cross(n0, n1);
  return std::atan2(std::sqrt(tf::dot(axis, axis)), tf::dot(n0, n1));
}

} // namespace tf::fill

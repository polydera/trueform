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

#include "../core/sidedness.hpp"
#include "./orient2d.hpp"
#include "./orient3d.hpp"
#include "./projection_axes.hpp"

namespace tf::exact {

/// Classify a point against a wedge formed by two faces sharing an edge.
///
/// The wedge is defined by:
///   - shared edge (v0, v1)
///   - face A = (v0, v1, a), face B = (v1, v0, b)
///   - test point pt
///
/// For CCW faces (outward normals):
///   on_negative_side = inside the wedge (below both planes)
///   on_positive_side = outside the wedge (above at least one plane)
///   on_boundary = on a face half-plane of the wedge
///
/// When pt is on a face's infinite plane (orient3d == 0), an orient2d
/// test checks that pt is on the correct half-plane (same side of the
/// edge as the face's opposite vertex, or on the edge itself).
///
/// Uses exact orient3d/orient2d — no floating point.
inline auto classify_wedge(const pt3 &v0, const pt3 &v1, const pt3 &a,
                           const pt3 &b, const pt3 &pt) -> tf::sidedness {
  auto sa = orient3d_value(v0, v1, a, pt);
  auto sb = orient3d_value(v1, v0, b, pt);

  auto on_half_plane = [](const pt3 &e0, const pt3 &e1, const pt3 &opp,
                          const pt3 &p) -> bool {
    auto axes = projection_axes(e0, e1, opp);
    int ax0 = axes.first, ax1 = axes.second;
    pt2 e0_2d = {e0[ax0], e0[ax1]};
    pt2 e1_2d = {e1[ax0], e1[ax1]};
    pt2 opp_2d = {opp[ax0], opp[ax1]};
    pt2 p_2d = {p[ax0], p[ax1]};
    auto s_opp = orient2d(e0_2d, e1_2d, opp_2d);
    auto s_p = orient2d(e0_2d, e1_2d, p_2d);
    if (s_p == 0) return true;
    return (s_p > 0) == (s_opp > 0);
  };

  if (sa == 0 && sb == 0)
    return tf::sidedness::on_boundary;
  else if (sa != 0 && sb != 0) {
    // Neither on a plane. Both same sign is unambiguous.
    if (sa < 0 && sb < 0)
      return tf::sidedness::on_negative_side;
    if (sa > 0 && sb > 0)
      return tf::sidedness::on_positive_side;
    // Mixed signs: need convex/reflex to resolve.
    // orient3d(v0, v1, a, b) > 0 → b above face A → convex opening,
    // solid interior is the reflex side → mixed = inside.
    auto wedge = orient3d_value(v0, v1, a, b);
    if (wedge > 0) {
      // Convex opening: solid interior is reflex. Mixed = inside.
      return tf::sidedness::on_negative_side;
    } else {
      // Reflex opening: solid interior is convex. Mixed = outside.
      return tf::sidedness::on_positive_side;
    }
  } else {
    // Exactly one is zero. Check if pt is on the correct half-plane.
    if (sa == 0) {
      if (on_half_plane(v0, v1, a, pt))
        return tf::sidedness::on_boundary;
      // Not on face A's half-plane — depends on sb.
      return sb > 0 ? tf::sidedness::on_positive_side
                     : tf::sidedness::on_negative_side;
    } else {
      if (on_half_plane(v1, v0, b, pt))
        return tf::sidedness::on_boundary;
      // Not on face B's half-plane — depends on sa.
      return sa > 0 ? tf::sidedness::on_positive_side
                     : tf::sidedness::on_negative_side;
    }
  }
}

} // namespace tf::exact

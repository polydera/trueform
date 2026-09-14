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
#include "../../core/cross.hpp"
#include "../../core/dot.hpp"
#include "../../core/point.hpp"
#include <cmath>

namespace tf::spatial {

/// The signed solid angle a triangle subtends at `q` — the exact
/// near-field term of the winding number (van Oosterom & Strackee).
/// A query on the triangle's own vertex or edge returns 0, not ±half —
/// atan2(0, den) is finite there, so an on-surface sample contributes
/// nothing rather than a NaN, and a signed distance built on it stays
/// safe at magnitude zero.
inline auto triangle_solid_angle(const tf::point<double, 3> &q,
                                 const tf::point<double, 3> &a,
                                 const tf::point<double, 3> &b,
                                 const tf::point<double, 3> &c) -> double {
  const auto ma = a - q;
  const auto mb = b - q;
  const auto mc = c - q;
  const auto la = ma.length();
  const auto lb = mb.length();
  const auto lc = mc.length();
  const auto det = tf::dot(ma, tf::cross(mb, mc));
  const auto den = la * lb * lc + tf::dot(ma, mb) * lc + tf::dot(mb, mc) * la +
                   tf::dot(mc, ma) * lb;
  return 2. * std::atan2(det, den);
}

} // namespace tf::spatial

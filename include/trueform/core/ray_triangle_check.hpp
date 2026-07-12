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
#include "./cross.hpp"
#include "./dot.hpp"
#include "./epsilon.hpp"
#include "./polygon.hpp"
#include "./ray_cast_info.hpp"
#include "./ray_config.hpp"
#include "./ray_like.hpp"
#include <limits>

namespace tf::core {

/// @brief Moller-Trumbore ray-triangle intersection test.
///
/// Two cross products, three dots, early-out on barycentric coordinates.
/// Replaces the generic plane + containment path for triangles.
// Cold path: parallel-vs-coplanar classification. Deliberately a
// separate function so the hot kernel below stays under the inliner's
// cost threshold -- the tree leaf calls it per candidate triangle.
template <typename RealT>
auto ray_triangle_degenerate_status(const RealT *n, RealT n_len2,
                                    const RealT *sv) -> tf::intersect_status {
  constexpr auto meps = std::numeric_limits<RealT>::epsilon();
  const RealT d = n[0] * sv[0] + n[1] * sv[1] + n[2] * sv[2];
  if (d * d < n_len2 * meps * meps)
    return tf::intersect_status::coplanar;
  return tf::intersect_status::parallel;
}

template <typename Policy0, typename Policy1>
inline auto ray_triangle_check(
    const tf::ray_like<3, Policy0> &ray,
    const tf::polygon<3, Policy1> &tri,
    const tf::ray_config<tf::coordinate_type<Policy0, Policy1>> &config) {
  using RealT = tf::coordinate_type<Policy0, Policy1>;
  constexpr auto eps = tf::epsilon<RealT>;
  constexpr auto meps = std::numeric_limits<RealT>::epsilon();

  // raw scalars on purpose: gathered once, flat float math -- the
  // vector-expression form measures ~25% slower in tree leaves
  RealT p0[3], e1[3], e2[3];
  {
    const auto v0 = tri[0];
    const auto v1 = tri[1];
    const auto v2 = tri[2];
    for (int d = 0; d < 3; ++d) {
      p0[d] = RealT(v0[d]);
      e1[d] = RealT(v1[d]) - p0[d];
      e2[d] = RealT(v2[d]) - p0[d];
    }
  }
  const RealT dx = RealT(ray.direction[0]);
  const RealT dy = RealT(ray.direction[1]);
  const RealT dz = RealT(ray.direction[2]);
  const RealT sx = RealT(ray.origin[0]) - p0[0];
  const RealT sy = RealT(ray.origin[1]) - p0[1];
  const RealT sz = RealT(ray.origin[2]) - p0[2];

  const RealT hx = dy * e2[2] - dz * e2[1];
  const RealT hy = dz * e2[0] - dx * e2[2];
  const RealT hz = dx * e2[1] - dy * e2[0];
  const RealT det = e1[0] * hx + e1[1] * hy + e1[2] * hz;

  const RealT nx = e1[1] * e2[2] - e1[2] * e2[1];
  const RealT ny = e1[2] * e2[0] - e1[0] * e2[2];
  const RealT nz = e1[0] * e2[1] - e1[1] * e2[0];
  const RealT n_len2 = nx * nx + ny * ny + nz * nz;
  const RealT d_len2 = dx * dx + dy * dy + dz * dz;
  if (det * det < d_len2 * n_len2 * meps) {
    const RealT n[3] = {nx, ny, nz};
    const RealT sv[3] = {sx, sy, sz};
    return tf::make_ray_cast_info(
        ray_triangle_degenerate_status(n, n_len2, sv), RealT{0});
  }

  const RealT inv_det = RealT{1} / det;
  const RealT u = inv_det * (sx * hx + sy * hy + sz * hz);
  if (u < -eps || u > RealT{1} + eps)
    return tf::make_ray_cast_info(tf::intersect_status::none, RealT{0});

  const RealT qx = sy * e1[2] - sz * e1[1];
  const RealT qy = sz * e1[0] - sx * e1[2];
  const RealT qz = sx * e1[1] - sy * e1[0];
  const RealT v = inv_det * (dx * qx + dy * qy + dz * qz);
  if (v < -eps || u + v > RealT{1} + eps)
    return tf::make_ray_cast_info(tf::intersect_status::none, RealT{0});

  const RealT t = inv_det * (e2[0] * qx + e2[1] * qy + e2[2] * qz);
  return tf::make_ray_cast_info(
      static_cast<tf::intersect_status>(
          char(t >= config.min_t - eps) &
          char(t <= config.max_t + eps)),
      t);
}

} // namespace tf::core

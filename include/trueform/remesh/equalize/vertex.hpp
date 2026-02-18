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

#include "../../core/sqrt.hpp"
#include "../../topology/half_edges.hpp"

namespace tf::remesh {

/// Compute the position of vertex v that minimizes the sum of squared
/// triangle areas around it. Computation in doubles for robustness.
/// Returns false if the system is singular or the 1-ring walk encounters
/// a non-manifold edge (vertex is not moved).
template <typename Index>
auto equalize_areas_vertex(const tf::half_edges<Index> &he,
                           const double *old_pos, Index v, double *out,
                           bool in_tangent_plane) -> bool {
  auto vhe = he.vertex_half_edge_handles()[v];
  double pv[3] = {old_pos[3 * v], old_pos[3 * v + 1], old_pos[3 * v + 2]};

  // Symmetric 3x3 matrix (upper triangle: xx,xy,xz,yy,yz,zz)
  double mat[6] = {};
  double rhs[3] = {};
  double norm[3] = {};

  auto cur = vhe;
  do {
    auto &h = he.half_edge(cur);
    if (h.is_simple()) {
      auto vi = he.end_vertex_handle(tf::unsafe, cur).id();
      auto vn =
          he.end_vertex_handle(tf::unsafe, he.next(tf::unsafe, cur)).id();
      double pi[3] = {old_pos[3 * vi], old_pos[3 * vi + 1],
                       old_pos[3 * vi + 2]};
      double pn[3] = {old_pos[3 * vn], old_pos[3 * vn + 1],
                       old_pos[3 * vn + 2]};
      // d = pn - pi (edge opposite to v in this face)
      double d[3] = {pn[0] - pi[0], pn[1] - pi[1], pn[2] - pi[2]};
      double len2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
      // crossSquare(d) = d*d^T - |d|^2*I
      double cs[6] = {d[0] * d[0] - len2, d[0] * d[1], d[0] * d[2],
                       d[1] * d[1] - len2, d[1] * d[2], d[2] * d[2] - len2};
      for (int i = 0; i < 6; ++i)
        mat[i] += cs[i];
      // rhs += crossSquare(d) * pi
      rhs[0] += cs[0] * pi[0] + cs[1] * pi[1] + cs[2] * pi[2];
      rhs[1] += cs[1] * pi[0] + cs[3] * pi[1] + cs[4] * pi[2];
      rhs[2] += cs[2] * pi[0] + cs[4] * pi[1] + cs[5] * pi[2];
      if (in_tangent_plane) {
        double e0[3] = {pi[0] - pv[0], pi[1] - pv[1], pi[2] - pv[2]};
        double e1[3] = {pn[0] - pv[0], pn[1] - pv[1], pn[2] - pv[2]};
        norm[0] += e0[1] * e1[2] - e0[2] * e1[1];
        norm[1] += e0[2] * e1[0] - e0[0] * e1[2];
        norm[2] += e0[0] * e1[1] - e0[1] * e1[0];
      }
    }
    cur = he.rotated(cur);
    if (!cur.is_valid())
      return false;
  } while (cur != vhe);

  constexpr double eps = std::numeric_limits<double>::epsilon();

  if (in_tangent_plane) {
    double nlen = tf::sqrt(norm[0] * norm[0] + norm[1] * norm[1] +
                           norm[2] * norm[2]);
    if (nlen < eps)
      return false;
    norm[0] /= nlen;
    norm[1] /= nlen;
    norm[2] /= nlen;
    // Tangent basis perpendicular to normal
    double t0[3], t1[3];
    if (std::abs(norm[0]) < std::abs(norm[1])) {
      t0[0] = 0;
      t0[1] = -norm[2];
      t0[2] = norm[1];
    } else {
      t0[0] = norm[2];
      t0[1] = 0;
      t0[2] = -norm[0];
    }
    double t0len =
        tf::sqrt(t0[0] * t0[0] + t0[1] * t0[1] + t0[2] * t0[2]);
    t0[0] /= t0len;
    t0[1] /= t0len;
    t0[2] /= t0len;
    t1[0] = norm[1] * t0[2] - norm[2] * t0[1];
    t1[1] = norm[2] * t0[0] - norm[0] * t0[2];
    t1[2] = norm[0] * t0[1] - norm[1] * t0[0];
    // Fixed normal component: p0 = dot(n, pv) * n
    double nd = norm[0] * pv[0] + norm[1] * pv[1] + norm[2] * pv[2];
    double p0[3] = {nd * norm[0], nd * norm[1], nd * norm[2]};
    // Project mat to 2x2 tangent-plane system
    double mt0[3] = {mat[0] * t0[0] + mat[1] * t0[1] + mat[2] * t0[2],
                     mat[1] * t0[0] + mat[3] * t0[1] + mat[4] * t0[2],
                     mat[2] * t0[0] + mat[4] * t0[1] + mat[5] * t0[2]};
    double mt1[3] = {mat[0] * t1[0] + mat[1] * t1[1] + mat[2] * t1[2],
                     mat[1] * t1[0] + mat[3] * t1[1] + mat[4] * t1[2],
                     mat[2] * t1[0] + mat[4] * t1[1] + mat[5] * t1[2]};
    double m2xx = mt0[0] * t0[0] + mt0[1] * t0[1] + mt0[2] * t0[2];
    double m2xy = mt0[0] * t1[0] + mt0[1] * t1[1] + mt0[2] * t1[2];
    double m2yy = mt1[0] * t1[0] + mt1[1] * t1[1] + mt1[2] * t1[2];
    // Adjusted rhs: rhs - mat*p0, projected onto tangent plane
    double mp0[3] = {mat[0] * p0[0] + mat[1] * p0[1] + mat[2] * p0[2],
                     mat[1] * p0[0] + mat[3] * p0[1] + mat[4] * p0[2],
                     mat[2] * p0[0] + mat[4] * p0[1] + mat[5] * p0[2]};
    double ar[3] = {rhs[0] - mp0[0], rhs[1] - mp0[1], rhs[2] - mp0[2]};
    double r2x = ar[0] * t0[0] + ar[1] * t0[1] + ar[2] * t0[2];
    double r2y = ar[0] * t1[0] + ar[1] * t1[1] + ar[2] * t1[2];
    // Solve 2x2
    double det2 = m2xx * m2yy - m2xy * m2xy;
    double tr2 = m2xx + m2yy;
    if (eps * std::abs(tr2 * tr2) >= std::abs(det2))
      return false;
    double inv2 = 1.0 / det2;
    double sx = inv2 * (m2yy * r2x - m2xy * r2y);
    double sy = inv2 * (m2xx * r2y - m2xy * r2x);
    out[0] = p0[0] + t0[0] * sx + t1[0] * sy;
    out[1] = p0[1] + t0[1] * sx + t1[1] * sy;
    out[2] = p0[2] + t0[2] * sx + t1[2] * sy;
  } else {
    // Full 3x3 Cramer's rule solve
    double det = mat[0] * (mat[3] * mat[5] - mat[4] * mat[4]) -
                 mat[1] * (mat[1] * mat[5] - mat[4] * mat[2]) +
                 mat[2] * (mat[1] * mat[4] - mat[3] * mat[2]);
    double tr = mat[0] + mat[3] + mat[5];
    if (eps * std::abs(tr * tr * tr) >= std::abs(det))
      return false;
    double inv = 1.0 / det;
    out[0] = inv * ((mat[3] * mat[5] - mat[4] * mat[4]) * rhs[0] +
                     (mat[2] * mat[4] - mat[1] * mat[5]) * rhs[1] +
                     (mat[1] * mat[4] - mat[2] * mat[3]) * rhs[2]);
    out[1] = inv * ((mat[2] * mat[4] - mat[1] * mat[5]) * rhs[0] +
                     (mat[0] * mat[5] - mat[2] * mat[2]) * rhs[1] +
                     (mat[1] * mat[2] - mat[0] * mat[4]) * rhs[2]);
    out[2] = inv * ((mat[1] * mat[4] - mat[2] * mat[3]) * rhs[0] +
                     (mat[1] * mat[2] - mat[0] * mat[4]) * rhs[1] +
                     (mat[0] * mat[3] - mat[1] * mat[1]) * rhs[2]);
  }
  return true;
}

} // namespace tf::remesh

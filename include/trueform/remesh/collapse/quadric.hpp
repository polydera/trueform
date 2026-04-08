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
#include <optional>

#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/algorithm/reduce.hpp"
#include "../../core/buffer.hpp"
#include "../../core/cross.hpp"
#include "../../core/eigen_of_symmetric.hpp"
#include "../../core/point.hpp"
#include "../../core/points.hpp"
#include "../../core/vector.hpp"
#include "../../core/views/mapped_range.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../core/views/zip.hpp"
#include "../../topology/half_edges.hpp"
#include "../feature_mask.hpp"

namespace tf::remesh {

/// @ingroup remesh
/// @brief Quadric error matrix (Garland & Heckbert).
///
/// Stores the upper triangle of a 3x3 symmetric matrix A, a linear
/// term b, and a constant c. The quadric error at point p is
/// p^T A p + 2 b^T p + c.
struct quadric {
  double A[6]; // upper triangle: A00,A01,A02,A11,A12,A22
  double b[3];
  double c;

  auto operator+=(const quadric &o) -> quadric & {
    for (int i = 0; i < 6; ++i)
      A[i] += o.A[i];
    for (int i = 0; i < 3; ++i)
      b[i] += o.b[i];
    c += o.c;
    return *this;
  }

  template <typename T>
  auto evaluate(const tf::point_like<3, T> &pt) const -> double {
    double p[3] = {double(pt[0]), double(pt[1]), double(pt[2])};
    double out = c;
    int ai = 0;
    for (int i = 0; i < 3; ++i) {
      out += p[i] * p[i] * A[ai++];
      out += 2.0 * b[i] * p[i];
      for (int j = i + 1; j < 3; ++j, ++ai)
        out += 2.0 * p[i] * p[j] * A[ai];
    }
    return out;
  }
};

/// @ingroup remesh
/// @brief Solve for the quadric-optimal point.
///
/// Tries Cramer's rule first (fast path). Falls back to
/// eigendecomposition-based pseudoinverse for ill-conditioned matrices.
/// Tikhonov regularization pulls the solution toward @p center in
/// degenerate directions: minimizes p^T A p + 2b^T p + λ||p - center||².
/// Returns std::nullopt if the matrix is degenerate.
template <typename Real>
auto solve_optimal_quadric(const quadric &q,
                           const tf::point<Real, 3> &center,
                           double stabilizer = 0)
    -> std::optional<tf::point<Real, 3>> {
  double a00 = q.A[0], a01 = q.A[1], a02 = q.A[2];
  double a11 = q.A[3], a12 = q.A[4];
  double a22 = q.A[5];
  // Relative singularity threshold: |det| vs Frobenius norm cubed
  double frob2 = a00 * a00 + a11 * a11 + a22 * a22 +
                 2.0 * (a01 * a01 + a02 * a02 + a12 * a12);
  double det = a00 * (a11 * a22 - a12 * a12) -
               a01 * (a01 * a22 - a12 * a02) +
               a02 * (a01 * a12 - a11 * a02);
  double frob3 = frob2 * tf::sqrt(frob2);
  if (std::abs(det) > 1e-8 * frob3) {
    // Well-conditioned: Cramer's rule (stabilizer not needed)
    double inv = 1.0 / det;
    double r0 = -q.b[0], r1 = -q.b[1], r2 = -q.b[2];
    return tf::make_point(
        Real(inv * ((a11 * a22 - a12 * a12) * r0 +
                     (a02 * a12 - a01 * a22) * r1 +
                     (a01 * a12 - a02 * a11) * r2)),
        Real(inv * ((a02 * a12 - a01 * a22) * r0 +
                     (a00 * a22 - a02 * a02) * r1 +
                     (a01 * a02 - a00 * a12) * r2)),
        Real(inv * ((a01 * a12 - a02 * a11) * r0 +
                     (a01 * a02 - a00 * a12) * r1 +
                     (a00 * a11 - a01 * a01) * r2)));
  }
  // Ill-conditioned: eigendecomposition with Tikhonov toward center
  // (A + λI)p = -b + λ*center
  std::array<std::array<double, 3>, 3> mat = {
      {{a00, a01, a02}, {a01, a11, a12}, {a02, a12, a22}}};
  auto [eigenvalues, eigenvectors] = tf::eigen_of_symmetric(mat);
  for (int i = 0; i < 3; ++i)
    eigenvalues[i] += stabilizer;
  double max_ev = std::max({std::abs(eigenvalues[0]),
                            std::abs(eigenvalues[1]),
                            std::abs(eigenvalues[2])});
  if (max_ev < std::numeric_limits<double>::epsilon())
    return std::nullopt;
  double threshold = max_ev * 1e-10;
  double r[3] = {-q.b[0] + stabilizer * double(center[0]),
                  -q.b[1] + stabilizer * double(center[1]),
                  -q.b[2] + stabilizer * double(center[2])};
  double result[3] = {0, 0, 0};
  int rank = 0;
  for (int i = 0; i < 3; ++i) {
    if (std::abs(eigenvalues[i]) <= threshold)
      continue;
    const auto &v = eigenvectors[i];
    double dot = double(v[0]) * r[0] + double(v[1]) * r[1] +
                 double(v[2]) * r[2];
    double coeff = dot / eigenvalues[i];
    result[0] += coeff * double(v[0]);
    result[1] += coeff * double(v[1]);
    result[2] += coeff * double(v[2]);
    ++rank;
  }
  if (rank == 0)
    return std::nullopt;
  return tf::make_point(Real(result[0]), Real(result[1]), Real(result[2]));
}

/// @ingroup remesh
/// @brief Solve for the quadric-optimal point via Cramer's rule.
///
/// Direct 3x3 solve using the determinant. Fast but numerically
/// unstable for rank-deficient quadrics. Returns std::nullopt if
/// the matrix is singular.
template <typename Real>
auto solve_optimal_quadric_direct(const quadric &q)
    -> std::optional<tf::point<Real, 3>> {
  double a00 = q.A[0], a01 = q.A[1], a02 = q.A[2];
  double a11 = q.A[3], a12 = q.A[4];
  double a22 = q.A[5];
  double det = a00 * (a11 * a22 - a12 * a12) -
               a01 * (a01 * a22 - a12 * a02) +
               a02 * (a01 * a12 - a11 * a02);
  if (std::abs(det) < std::numeric_limits<double>::epsilon())
    return std::nullopt;
  double inv = 1.0 / det;
  double r0 = -q.b[0], r1 = -q.b[1], r2 = -q.b[2];
  return tf::make_point(Real(inv * ((a11 * a22 - a12 * a12) * r0 +
                                    (a02 * a12 - a01 * a22) * r1 +
                                    (a01 * a12 - a02 * a11) * r2)),
                        Real(inv * ((a02 * a12 - a01 * a22) * r0 +
                                    (a00 * a22 - a02 * a02) * r1 +
                                    (a01 * a02 - a00 * a12) * r2)),
                        Real(inv * ((a01 * a12 - a02 * a11) * r0 +
                                    (a01 * a02 - a00 * a12) * r1 +
                                    (a00 * a11 - a01 * a01) * r2)));
}

/// @ingroup remesh
/// @brief Compute per-vertex quadric error matrices.
///
/// Builds one quadric per face (from the face plane equation), then
/// sums face quadrics into per-vertex quadrics.
template <typename Index, typename PointsPolicy>
auto compute_vertex_quadrics(const tf::half_edges<Index> &he,
                             const tf::points<PointsPolicy> &points)
    -> tf::buffer<quadric> {
  Index n_faces = Index(he.face_half_edge_handles().size());
  Index n_verts = Index(he.vertex_half_edge_handles().size());

  tf::buffer<quadric> face_quadrics;
  face_quadrics.allocate(n_faces);

  tf::parallel_for_each(
      tf::zip(he.face_half_edge_handles(), face_quadrics), [&](auto tup) {
        auto &&[fhe, fq] = tup;
        if (!fhe.is_valid()) {
          fq = {};
          return;
        }
        auto h0 = fhe;
        auto h1 = he.next(tf::unsafe, h0);
        auto h2 = he.next(tf::unsafe, h1);
        auto p0 = points[he.start_vertex_handle(tf::unsafe, h0).id()];
        auto p1 = points[he.start_vertex_handle(tf::unsafe, h1).id()];
        auto p2 = points[he.start_vertex_handle(tf::unsafe, h2).id()];
        auto e0 =
            tf::make_vector(double(p1[0] - p0[0]), double(p1[1] - p0[1]),
                            double(p1[2] - p0[2]));
        auto e1 =
            tf::make_vector(double(p2[0] - p0[0]), double(p2[1] - p0[1]),
                            double(p2[2] - p0[2]));
        auto n = tf::cross(e0, e1);
        auto len = n.length();
        if (len < std::numeric_limits<double>::epsilon()) {
          fq = {};
          return;
        }
        n[0] /= len;
        n[1] /= len;
        n[2] /= len;
        double d = -(n[0] * double(p0[0]) + n[1] * double(p0[1]) +
                     n[2] * double(p0[2]));
        int ai = 0;
        for (int i = 0; i < 3; ++i)
          for (int j = i; j < 3; ++j, ++ai)
            fq.A[ai] = n[i] * n[j];
        fq.b[0] = d * n[0];
        fq.b[1] = d * n[1];
        fq.b[2] = d * n[2];
        fq.c = d * d;
      });

  tf::buffer<quadric> vertex_quadrics;
  vertex_quadrics.allocate(n_verts);
  tf::parallel_for_each(
      tf::zip(he.vertex_half_edge_handles(), vertex_quadrics), [&](auto tup) {
        auto &&[vhe, vq] = tup;
        vq = {};
        if (!vhe.is_valid())
          return;
        auto cur = vhe;
        do {
          auto fh = he.face_handle(tf::unsafe, cur).id();
          if (fh >= 0)
            vq += face_quadrics[fh];
          cur = he.rotated(cur);
          if (!cur.is_valid())
            break;
        } while (cur != vhe);
      });

  return vertex_quadrics;
}

/// @ingroup remesh
/// @brief Compute per-vertex quadric error matrices with feature penalties.
///
/// Same as the base overload, but also stores face normals and adds
/// weighted constraint-plane quadrics for feature edges during the
/// vertex accumulation pass. No extra passes over edges.
///
/// For each feature edge incident to a vertex, two perpendicular
/// constraint planes (one per adjacent face) are added with weight
/// @p feature_weight * mean_vertex_quadric_trace.
///
/// @param mask Feature mask. If empty, equivalent to the base overload.
/// @param feature_weight Penalty multiplier for feature quadrics.
template <typename Index, typename PointsPolicy>
auto compute_vertex_quadrics(const tf::half_edges<Index> &he,
                             const tf::points<PointsPolicy> &points,
                             const tf::remesh::feature_mask &mask,
                             double feature_weight)
    -> tf::buffer<quadric> {
  if (mask.empty())
    return compute_vertex_quadrics(he, points);

  Index n_faces = Index(he.face_half_edge_handles().size());
  Index n_verts = Index(he.vertex_half_edge_handles().size());

  // Pass 1: face quadrics + face normals (parallel)
  tf::buffer<quadric> face_quadrics;
  face_quadrics.allocate(n_faces);
  tf::buffer<double> face_normals;
  face_normals.allocate(n_faces * 3);

  tf::parallel_for_each(
      tf::make_sequence_range(n_faces), [&](Index fi) {
        auto fhe = he.face_half_edge_handles()[fi];
        if (!fhe.is_valid()) {
          face_quadrics[fi] = {};
          face_normals[fi * 3 + 0] = 0;
          face_normals[fi * 3 + 1] = 0;
          face_normals[fi * 3 + 2] = 0;
          return;
        }
        auto h0 = fhe;
        auto h1 = he.next(tf::unsafe, h0);
        auto h2 = he.next(tf::unsafe, h1);
        auto p0 = points[he.start_vertex_handle(tf::unsafe, h0).id()];
        auto p1 = points[he.start_vertex_handle(tf::unsafe, h1).id()];
        auto p2 = points[he.start_vertex_handle(tf::unsafe, h2).id()];
        auto e0 =
            tf::make_vector(double(p1[0] - p0[0]), double(p1[1] - p0[1]),
                            double(p1[2] - p0[2]));
        auto e1 =
            tf::make_vector(double(p2[0] - p0[0]), double(p2[1] - p0[1]),
                            double(p2[2] - p0[2]));
        auto n = tf::cross(e0, e1);
        auto len = n.length();
        if (len < std::numeric_limits<double>::epsilon()) {
          face_quadrics[fi] = {};
          face_normals[fi * 3 + 0] = 0;
          face_normals[fi * 3 + 1] = 0;
          face_normals[fi * 3 + 2] = 0;
          return;
        }
        n[0] /= len;
        n[1] /= len;
        n[2] /= len;
        face_normals[fi * 3 + 0] = n[0];
        face_normals[fi * 3 + 1] = n[1];
        face_normals[fi * 3 + 2] = n[2];
        double d = -(n[0] * double(p0[0]) + n[1] * double(p0[1]) +
                     n[2] * double(p0[2]));
        auto &fq = face_quadrics[fi];
        int ai = 0;
        for (int i = 0; i < 3; ++i)
          for (int j = i; j < 3; ++j, ++ai)
            fq.A[ai] = n[i] * n[j];
        fq.b[0] = d * n[0];
        fq.b[1] = d * n[1];
        fq.b[2] = d * n[2];
        fq.c = d * d;
      });

  // Compute mean face quadric trace for penalty scaling
  auto trace_sum = tf::reduce(
      tf::make_mapped_range(tf::make_range(face_quadrics),
                            [](const quadric &fq) {
                              return fq.A[0] + fq.A[3] + fq.A[5];
                            }),
      [](double acc, double tr) { return acc + tr; }, 0.0);
  double w = (n_faces > 0) ? feature_weight * trace_sum / n_faces : 0;

  // Pass 2: accumulate face quadrics + feature penalty quadrics per vertex
  tf::buffer<quadric> vertex_quadrics;
  vertex_quadrics.allocate(n_verts);

  tf::parallel_for_each(
      tf::make_sequence_range(n_verts), [&](Index vi) {
        auto &vq = vertex_quadrics[vi];
        vq = {};
        auto vhe = he.vertex_half_edge_handles()[vi];
        if (!vhe.is_valid())
          return;
        auto cur = vhe;
        do {
          // Sum face quadric
          if (he.is_simple(tf::unsafe, cur))
            vq += face_quadrics[he.face_handle(tf::unsafe, cur).id()];

          // Check if this edge is a feature edge
          if (w > 0) {
            auto eid = he.edge_handle(tf::unsafe, cur).id();
            if (mask.is_feature(eid)) {
              auto opp = he.opposite(tf::unsafe, cur);
              if (he.is_simple(tf::unsafe, cur) &&
                  he.is_simple(tf::unsafe, opp)) {
                auto f0 = he.face_handle(tf::unsafe, cur).id();
                auto f1 = he.face_handle(tf::unsafe, opp).id();
                // Edge direction
                auto v0 = he.start_vertex_handle(tf::unsafe, cur).id();
                auto v1 = he.end_vertex_handle(tf::unsafe, cur).id();
                double ed[3] = {double(points[v1][0]) - double(points[v0][0]),
                                double(points[v1][1]) - double(points[v0][1]),
                                double(points[v1][2]) - double(points[v0][2])};
                double elen2 = ed[0] * ed[0] + ed[1] * ed[1] + ed[2] * ed[2];
                if (elen2 < 1e-60) {
                  cur = he.rotated(cur);
                  if (!cur.is_valid())
                    break;
                  continue;
                }
                double inv_elen = 1.0 / tf::sqrt(elen2);
                ed[0] *= inv_elen;
                ed[1] *= inv_elen;
                ed[2] *= inv_elen;

                double pv[3] = {double(points[vi][0]), double(points[vi][1]),
                                double(points[vi][2])};

                // Add constraint plane for each adjacent face
                for (Index fi : {Index(f0), Index(f1)}) {
                  double fn[3] = {face_normals[fi * 3 + 0],
                                  face_normals[fi * 3 + 1],
                                  face_normals[fi * 3 + 2]};
                  // Constraint normal: edge_dir x face_normal
                  double nc[3] = {ed[1] * fn[2] - ed[2] * fn[1],
                                  ed[2] * fn[0] - ed[0] * fn[2],
                                  ed[0] * fn[1] - ed[1] * fn[0]};
                  double nc2 = nc[0] * nc[0] + nc[1] * nc[1] + nc[2] * nc[2];
                  if (nc2 < 1e-60)
                    continue;
                  double inv_nc = 1.0 / tf::sqrt(nc2);
                  nc[0] *= inv_nc;
                  nc[1] *= inv_nc;
                  nc[2] *= inv_nc;

                  double d = -(nc[0] * pv[0] + nc[1] * pv[1] + nc[2] * pv[2]);

                  quadric q{};
                  int ai = 0;
                  for (int i = 0; i < 3; ++i)
                    for (int j = i; j < 3; ++j, ++ai)
                      q.A[ai] = w * nc[i] * nc[j];
                  q.b[0] = w * d * nc[0];
                  q.b[1] = w * d * nc[1];
                  q.b[2] = w * d * nc[2];
                  q.c = w * d * d;
                  vq += q;
                }
              }
            }
          }
          cur = he.rotated(cur);
          if (!cur.is_valid())
            break;
        } while (cur != vhe);
      });

  return vertex_quadrics;
}

/// @ingroup remesh
/// @brief Compute the quadric collapse error for an edge.
///
/// Combines the vertex quadrics, solves for the optimal point, and
/// returns the square root of the quadric error at that point.
/// For boundary vertices, evaluates at the surviving vertex instead.
template <typename Real, typename Index, typename PointsPolicy>
auto collapse_error_quadric(const tf::buffer<quadric> &quadrics,
                            const tf::points<PointsPolicy> &points,
                            const tf::half_edges<Index> &he,
                            tf::half_edge_handle<Index> heh,
                            double stabilizer = 0) -> Real {
  auto v0 = he.start_vertex_handle(tf::unsafe, heh).id();
  auto v1 = he.end_vertex_handle(tf::unsafe, heh).id();
  quadric q = quadrics[v0];
  q += quadrics[v1];
  if (he.is_boundary_vertex(v0) || he.is_boundary_vertex(v1))
    return tf::sqrt(std::max(Real(q.evaluate(points[v0])), Real(0)));
  auto mid = tf::make_point(Real(points[v0][0] + points[v1][0]) / 2,
                            Real(points[v0][1] + points[v1][1]) / 2,
                            Real(points[v0][2] + points[v1][2]) / 2);
  if (auto opt = solve_optimal_quadric<Real>(q, mid, stabilizer))
    return tf::sqrt(std::max(Real(q.evaluate(*opt)), Real(0)));
  auto e0 = q.evaluate(points[v0]);
  auto e1 = q.evaluate(points[v1]);
  return tf::sqrt(std::max(Real(std::min({e0, e1, q.evaluate(mid)})), Real(0)));
}

/// @ingroup remesh
/// @brief Compute the quadric-optimal collapse point.
///
/// Tries pseudoinverse solve first; falls back to the best of v0, v1, or
/// their midpoint (whichever minimizes the combined quadric error).
template <typename Real, typename Index, typename PointsPolicy>
auto collapsed_point_quadric(const tf::buffer<quadric> &quadrics,
                             const tf::points<PointsPolicy> &points, Index v0,
                             Index v1, double stabilizer = 0)
    -> tf::point<Real, 3> {
  quadric q = quadrics[v0];
  q += quadrics[v1];
  auto mid = tf::make_point(Real(points[v0][0] + points[v1][0]) / 2,
                            Real(points[v0][1] + points[v1][1]) / 2,
                            Real(points[v0][2] + points[v1][2]) / 2);
  if (auto opt = solve_optimal_quadric<Real>(q, mid, stabilizer))
    return *opt;
  auto p0 = tf::make_point(Real(points[v0][0]), Real(points[v0][1]),
                            Real(points[v0][2]));
  auto p1 = tf::make_point(Real(points[v1][0]), Real(points[v1][1]),
                            Real(points[v1][2]));
  auto e0 = std::make_pair(q.evaluate(p0), p0);
  auto e1 = std::make_pair(q.evaluate(p1), p1);
  auto em = std::make_pair(q.evaluate(mid), mid);
  return std::min({e0, e1, em}).second;
}

/// @ingroup remesh
/// @brief Compute the quadric-optimal collapse point via Cramer's rule.
///
/// Tries direct determinant solve first; falls back to the best of v0, v1, or
/// their midpoint (whichever minimizes the combined quadric error).
template <typename Real, typename Index, typename PointsPolicy>
auto collapsed_point_quadric_direct(const tf::buffer<quadric> &quadrics,
                                    const tf::points<PointsPolicy> &points,
                                    Index v0, Index v1) -> tf::point<Real, 3> {
  quadric q = quadrics[v0];
  q += quadrics[v1];
  if (auto opt = solve_optimal_quadric_direct<Real>(q))
    return *opt;
  auto p0 = tf::make_point(Real(points[v0][0]), Real(points[v0][1]),
                            Real(points[v0][2]));
  auto p1 = tf::make_point(Real(points[v1][0]), Real(points[v1][1]),
                            Real(points[v1][2]));
  auto mid = tf::make_point((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2,
                            (p0[2] + p1[2]) / 2);
  auto e0 = std::make_pair(q.evaluate(p0), p0);
  auto e1 = std::make_pair(q.evaluate(p1), p1);
  auto em = std::make_pair(q.evaluate(mid), mid);
  return std::min({e0, e1, em}).second;
}

/// @ingroup remesh
/// @brief Commit a quadric collapse: merge quadrics and update position.
template <typename Real, typename Index, typename PointsPolicy>
auto commit_collapse_quadric(tf::buffer<quadric> &quadrics, Index v0, Index v1,
                             const tf::point<Real, 3> &pt,
                             tf::points<PointsPolicy> &points) -> void {
  quadrics[v0] += quadrics[v1];
  for (int d = 0; d < 3; ++d)
    points[v0][d] = pt[d];
}

} // namespace tf::remesh

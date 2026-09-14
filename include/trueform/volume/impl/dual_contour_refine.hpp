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
#include "../../core/algorithm/generic_generate.hpp"
#include "../../core/algorithm/parallel_for_each.hpp"
#include "../../core/buffer.hpp"
#include "../../core/covariance_of.hpp"
#include "../../core/eigen_of_symmetric.hpp"
#include "../../core/point.hpp"
#include "../../core/points.hpp"
#include "../../core/range.hpp"
#include "../../core/vector.hpp"
#include "../../core/vectors.hpp"
#include "../../core/views/sequence_range.hpp"
#include "../../remesh/collapse/quadric.hpp"
#include "./dual_contour_cells.hpp"
#include "./dual_contour_planes.hpp"
#include "./dual_contour_tables.hpp"
#include "./dual_contour_vertices.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tf {
namespace volume_detail {

// Normal-covariance trace below this is one plane: the rank-1 case, which the
// vertex pass already places correctly.
inline constexpr double k_face_spread = 1e-2;

// Two fitted planes this close to facing each other bound a thin sheet, not a
// crease: there is no edge between them to recover.
inline constexpr double k_sheet_dot = -0.95;

// A pooled normal covariance this close to rank one lies about a single axis:
// the population is two clusters and no more, which is the premise the mean
// below is read under. Three faces around a corner leave rank two and are
// refused here.
inline constexpr double k_sheet_rank = 0.2;

// Two clusters an angle `a` apart leave a mean of length `cos(a/2)`, so a
// squared mean this small is `a > 172` degrees — ten degrees past the 162 that
// @ref k_sheet_dot draws, which is the margin between what the normals state
// and what a plane fitted through their crossings would.
inline constexpr double k_sheet_mean = 5e-3;

// What a certified solve pays for the direction its quadric leaves free. It is
// not insurance against the planes — those are the field's own — only the
// choice of a point along the crease they meet in.
inline constexpr double k_certified_stabilizer = 1e-6;

// Cells of escape a certified solve is allowed, against the one an estimated
// solve gets: a shallow apex genuinely sits that far outside its own cell.
inline constexpr int k_certified_escape = 3;

// How far past a sample's own reach a certified vertex may sit, in cells,
// before the samples are taken to refuse it.
inline constexpr double k_tangency_band = 0.05;

/// @brief What the refinement did with a vertex, and why.
enum class refit_state : unsigned char {
  unrefined,   ///< the feature gate did not flag its cell
  no_pool,     ///< flagged, but the neighbourhood states too few crossings
  one_face,    ///< the pooled normals state a single plane
  sheet,       ///< two fitted planes face each other: no edge to recover
  no_feature,  ///< the component lies on fewer than two certified planes
  off_surface, ///< the samples refuse a surface point where the planes meet
  clamped,     ///< the fitted planes solved outside the cells allowed
  refit        ///< the vertex is the fitted planes' own solution
};

/// @brief Per vertex, what the refinement did with it. Written only when the
/// builder was asked for it.
struct refit_record {
  refit_state state = refit_state::unrefined;
  unsigned char planes = 0; ///< how many fitted planes its quadric read
  bool certified = false;   ///< whether the samples stated those planes exactly
  float moved = 0.f;        ///< how far the refit moved it, in world units
};

/// @brief Refit the vertices of cells whose crossings state more than one
/// plane, from the planes their neighbourhood actually lies on.
///
/// The gate is the rank of each component's own normal covariance, whose trace
/// for unit vectors is `1 - |mean|^2`: one plane leaves no spread, a crease or
/// corner does. A sheet — two fitted planes facing each other — has no edge to
/// recover and keeps the vertex it has.
///
/// A flagged cell pools its 3x3x3 neighbourhood's crossings, clusters them by
/// normal, and fits one plane per cluster through its points. That much states
/// the planes for ANY field, and it is where the pass ends for one that is not
/// distance-like.
///
/// For one that IS, those planes are only a seed. A distance field is exactly
/// `n . p + d` inside a face's nearest-feature region, so the pool's SAMPLES
/// determine the planes outright — no estimated gradient, no interpolated
/// crossing — and a sample the model does not explain rejects itself through
/// its own residual. When two planes come out certified against their own
/// samples the cell is solved from them: without the stabilizer's tax, allowed
/// further out of its cell, and admitted only if the samples leave room for a
/// surface point where the planes meet. That last test is what tells a crease
/// from a fillet, whose two flanks are just as certified and whose meet is off
/// the surface by the radius it rounds.
///
/// One shot per cell, no feedback, so it is deterministic and parallel.
///
/// @param provenance When the builder asked for it, receives per vertex what
///        this pass did with it. Unread otherwise.
template <bool Record = false, typename Index, typename Points, typename Real,
          typename Samples>
auto refine_feature_vertices(const dual_contour_cells<Index> &cells,
                             const Samples &samples, Real iso,
                             double stabilizer, Index vertex_count,
                             const float *normal_spread,
                             tf::buffer<std::ptrdiff_t> &flagged, Points points,
                             refit_record *provenance = nullptr) -> void {
  using Compute = double;
  const int cx = cells.g.cx, cy = cells.g.cy, cz = cells.g.cz;
  const auto cell_rows = static_cast<std::ptrdiff_t>(cy) * cz;
  {
    // A component's crossing normals are unit vectors, so their
    // covariance trace is exactly 1 - |mean|^2 — the rank-1 test without
    // a decomposition. One plane leaves no spread; a crease or corner
    // does. Per component, since two sheets of a thin slab are two
    // components, each flat.
    flagged.clear();
    tf::generic_generate(
        tf::make_sequence_range(cell_rows), flagged,
        [&](std::ptrdiff_t row, auto &outflagged) {
          const int z = int(row / cy), y = int(row % cy);
          const auto active_end = cells.active_offsets[row + 1];
          const Index row_vertex_end = row + 1 < cell_rows
                                           ? cells.row_meta[(row + 1) * 8]
                                           : vertex_count;
          for (auto entry = cells.active_offsets[row]; entry < active_end;
               ++entry) {
            const Index base = cells.active_base[entry];
            const Index next = entry + 1 < active_end
                                   ? cells.active_base[entry + 1]
                                   : row_vertex_end;
            for (Index id = base; id < next; ++id) {
              if (normal_spread[id] <= k_face_spread)
                continue;
              outflagged.push_back((static_cast<std::ptrdiff_t>(z) * cy + y) *
                                       cx +
                                   cells.active_x[entry]);
              break;
            }
          }
        });
    tf::parallel_for_each(
        tf::make_sequence_range(std::ptrdiff_t(flagged.size())),
        [&](std::ptrdiff_t flagged_id) {
          const std::ptrdiff_t linear = flagged[flagged_id];
          int x = int(linear % cx);
          int y = int((linear / cx) % cy);
          int z = int(linear / (std::ptrdiff_t(cx) * cy));
          const Index own_base = cells.base_of(x, y, z);
          const unsigned own_retained = cells.retained_at(x, y, z);
          const auto own_case =
              cell_case(&cells.x_cases[cells.g.x_case_row(y, z) + x],
                        &cells.x_cases[cells.g.x_case_row(y + 1, z) + x],
                        &cells.x_cases[cells.g.x_case_row(y, z + 1) + x],
                        &cells.x_cases[cells.g.x_case_row(y + 1, z + 1) + x]);
          auto state_cell = [&](refit_state state) {
            if constexpr (Record) {
              const int own_vertices = cell_vertex_count(own_case, own_retained);
              for (int c = 0; c < own_vertices; ++c)
                provenance[own_base + Index(c)].state = state;
            } else
              static_cast<void>(state);
          };
          const int k_pool_cap = 224;
          tf::point<double, 3> pool_p[k_pool_cap];
          tf::vector<double, 3> pool_n[k_pool_cap];
          {
            int n_pool = 0;
            int own_comp_begin[4] = {-1, -1, -1, -1};
            int own_comp_end[4] = {-1, -1, -1, -1};
            int own_n_comp = 0;
            // the pool's own grid points: the twenty-seven cells meet at
            // sixty-four of them, and every crossing of every cell reads its
            // gradient there
            pool_gradients<Compute, Samples> gradients(cells.g, samples, x - 1,
                                                       y - 1, z - 1);
            for (int dz = -1; dz <= 1; ++dz)
              for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                  int nx_ = x + dx, ny_ = y + dy, nz_ = z + dz;
                  if (nx_ < 0 || ny_ < 0 || nz_ < 0 || nx_ >= cx || ny_ >= cy ||
                      nz_ >= cz)
                    continue;
                  auto nid = cells.base_of(nx_, ny_, nz_);
                  if (nid < 0)
                    continue;
                  const auto composite = cell_case(
                      &cells.x_cases[cells.g.x_case_row(ny_, nz_) + nx_],
                      &cells.x_cases[cells.g.x_case_row(ny_ + 1, nz_) + nx_],
                      &cells.x_cases[cells.g.x_case_row(ny_, nz_ + 1) + nx_],
                      &cells.x_cases[cells.g.x_case_row(ny_ + 1, nz_ + 1) +
                                     nx_]);
                  int n_comp = components().count[composite];
                  bool own = dx == 0 && dy == 0 && dz == 0;
                  if (own)
                    own_n_comp = std::min(n_comp, 4);
                  for (int comp = 0; comp < n_comp; ++comp) {
                    if (own && comp < 4)
                      own_comp_begin[comp] = n_pool;
                    std::array<double, 3> pos[12], nrm[12];
                    const int m = cell_crossings<Compute>(
                        cells.g, samples, Compute(iso), nx_, ny_, nz_,
                        composite, comp, gradients, pos, nrm, 12);
                    for (int k = 0; k < m && n_pool < k_pool_cap; ++k) {
                      // a crossing that states no plane cannot join a
                      // cluster of planes
                      if (nrm[k][0] == 0.0 && nrm[k][1] == 0.0 &&
                          nrm[k][2] == 0.0)
                        continue;
                      pool_p[n_pool] =
                          tf::make_point(pos[k][0], pos[k][1], pos[k][2]);
                      pool_n[n_pool] =
                          tf::make_vector(nrm[k][0], nrm[k][1], nrm[k][2]);
                      ++n_pool;
                    }
                    if (own && comp < 4)
                      own_comp_end[comp] = n_pool;
                  }
                }
            int own_begin = own_n_comp > 0 ? own_comp_begin[0] : -1;
            int own_end = own_n_comp > 0 ? own_comp_end[own_n_comp - 1] : -1;
            if (own_end - own_begin < 2 || n_pool < 4) {
              state_cell(refit_state::no_pool);
              return;
            }

            // spread of the pooled normals decides face / crease / corner
            auto normals =
                tf::make_vectors(tf::make_range(pool_n, pool_n + n_pool));
            auto [n_mean, n_cov] = tf::covariance_of(normals);
            auto [n_eval, n_evec] = tf::eigen_of_symmetric(n_cov);
            if (n_eval[2] < 1e-3) {
              state_cell(refit_state::one_face);
              return; // one face: the vertex pass already placed it
            }
            // Normals spread about one axis with a mean of nothing point
            // both ways in equal number — a sheet before a single plane
            // is fitted.
            const double n_mean2 = n_mean[0] * n_mean[0] +
                                   n_mean[1] * n_mean[1] +
                                   n_mean[2] * n_mean[2];
            if (n_eval[1] < k_sheet_rank * n_eval[2] &&
                n_mean2 < k_sheet_mean) {
              state_cell(refit_state::sheet);
              return;
            }
            const int k = n_eval[1] > 0.2 * n_eval[2] ? 3 : 2;

            // farthest-point seeding + a few spherical k-means rounds
            double seed[3][3];
            {
              int s0 = 0;
              double worst = 2.0;
              for (int i = 0; i < n_pool; ++i) {
                double d = pool_n[i][0] * n_mean[0] + pool_n[i][1] * n_mean[1] +
                           pool_n[i][2] * n_mean[2];
                if (d < worst) {
                  worst = d;
                  s0 = i;
                }
              }
              int s1 = 0;
              worst = 2.0;
              for (int i = 0; i < n_pool; ++i) {
                double d = pool_n[i][0] * pool_n[s0][0] +
                           pool_n[i][1] * pool_n[s0][1] +
                           pool_n[i][2] * pool_n[s0][2];
                if (d < worst) {
                  worst = d;
                  s1 = i;
                }
              }
              int s2 = 0;
              worst = 2.0;
              for (int i = 0; i < n_pool; ++i) {
                double d = std::max(pool_n[i][0] * pool_n[s0][0] +
                                        pool_n[i][1] * pool_n[s0][1] +
                                        pool_n[i][2] * pool_n[s0][2],
                                    pool_n[i][0] * pool_n[s1][0] +
                                        pool_n[i][1] * pool_n[s1][1] +
                                        pool_n[i][2] * pool_n[s1][2]);
                if (d < worst) {
                  worst = d;
                  s2 = i;
                }
              }
              const int si[3] = {s0, s1, s2};
              for (int j = 0; j < 3; ++j)
                for (int d = 0; d < 3; ++d)
                  seed[j][d] = pool_n[si[j]][d];
            }
            unsigned char assign[k_pool_cap];
            for (int round = 0; round < 3; ++round) {
              for (int i = 0; i < n_pool; ++i) {
                int best = 0;
                double best_d = -2.0;
                for (int j = 0; j < k; ++j) {
                  double d = pool_n[i][0] * seed[j][0] +
                             pool_n[i][1] * seed[j][1] +
                             pool_n[i][2] * seed[j][2];
                  if (d > best_d) {
                    best_d = d;
                    best = j;
                  }
                }
                assign[i] = static_cast<unsigned char>(best);
              }
              for (int j = 0; j < k; ++j) {
                double acc[3] = {0, 0, 0};
                for (int i = 0; i < n_pool; ++i)
                  if (assign[i] == j)
                    for (int d = 0; d < 3; ++d)
                      acc[d] += pool_n[i][d];
                double l = std::sqrt(acc[0] * acc[0] + acc[1] * acc[1] +
                                     acc[2] * acc[2]);
                if (l > 1e-30)
                  for (int d = 0; d < 3; ++d)
                    seed[j][d] = acc[d] / l;
              }
            }

            // one plane per cluster: PCA, normal = smallest-eigenvalue
            // direction, one reassignment round by point-plane distance
            double plane_n[k_plane_cap][3]{}, plane_d[k_plane_cap]{};
            bool plane_ok[k_plane_cap] = {false, false, false};
            for (int refit = 0; refit < 2; ++refit) {
              for (int j = 0; j < k; ++j) {
                tf::point<double, 3> cluster_p[k_pool_cap];
                int m = 0;
                double mean_n[3] = {0, 0, 0};
                for (int i = 0; i < n_pool; ++i)
                  if (assign[i] == j) {
                    cluster_p[m++] = pool_p[i];
                    for (int d = 0; d < 3; ++d)
                      mean_n[d] += pool_n[i][d];
                  }
                plane_ok[j] = m >= 3;
                if (!plane_ok[j])
                  continue;
                auto cluster_points =
                    tf::make_points(tf::make_range(cluster_p, cluster_p + m));
                auto [centroid, cov] = tf::covariance_of(cluster_points);
                auto [evals, evecs] = tf::eigen_of_symmetric(cov);
                double n[3] = {double(evecs[0][0]), double(evecs[0][1]),
                               double(evecs[0][2])};
                double sign =
                    n[0] * mean_n[0] + n[1] * mean_n[1] + n[2] * mean_n[2] < 0
                        ? -1.0
                        : 1.0;
                for (auto &v : n)
                  v *= sign;
                plane_n[j][0] = n[0];
                plane_n[j][1] = n[1];
                plane_n[j][2] = n[2];
                plane_d[j] =
                    -(n[0] * double(centroid[0]) + n[1] * double(centroid[1]) +
                      n[2] * double(centroid[2]));
              }
              if (refit == 1)
                break;
              for (int i = 0; i < n_pool; ++i) {
                int best = assign[i];
                double best_d = 1e30;
                for (int j = 0; j < k; ++j) {
                  if (!plane_ok[j])
                    continue;
                  double d =
                      std::abs(plane_n[j][0] * pool_p[i][0] +
                               plane_n[j][1] * pool_p[i][1] +
                               plane_n[j][2] * pool_p[i][2] + plane_d[j]);
                  if (d < best_d) {
                    best_d = d;
                    best = j;
                  }
                }
                assign[i] = static_cast<unsigned char>(best);
              }
            }

            // planes that face each other are the two sides of a sheet, not
            // a crease: nothing to sharpen, the pass-4 vertex stands
            for (int j0 = 0; j0 < k; ++j0)
              for (int j1 = j0 + 1; j1 < k; ++j1)
                if (plane_ok[j0] && plane_ok[j1] &&
                    plane_n[j0][0] * plane_n[j1][0] +
                            plane_n[j0][1] * plane_n[j1][1] +
                            plane_n[j0][2] * plane_n[j1][2] <
                        k_sheet_dot) {
                  state_cell(refit_state::sheet);
                  return;
                }

            // The clustered planes are a seed. Where the field is distance-like
            // they are only close, and the samples themselves state the planes
            // exactly: regress each on the samples it explains, and let a
            // sample the model does not explain reject itself.
            tf::point<double, 3> sample_p[k_pool_sample_cap];
            double sample_v[k_pool_sample_cap];
            const int n_samples = gather_pool_samples<Compute>(
                cells.g, samples, Compute(iso), x, y, z, sample_p, sample_v);
            pool_plane fitted[k_plane_cap]{};
            const double h = (cells.g.spacing[0] + cells.g.spacing[1] +
                              cells.g.spacing[2]) /
                             3.0;
            const bool certified =
                state_pool_planes(sample_p, sample_v, n_samples, pool_p, pool_n,
                                  n_pool, h, plane_n, plane_d, plane_ok,
                                  fitted) >= 2;
            double gradient = 1.0;
            if (certified) {
              // the certified planes replace their seeds, and the pool is
              // reassigned to them with a reject class: a crossing that lies on
              // none of them states no plane for the quadric to read
              double grad_sum = 0;
              int grad_n = 0;
              for (int j = 0; j < k_plane_cap; ++j) {
                plane_ok[j] = fitted[j].certified;
                if (!plane_ok[j])
                  continue;
                for (int d = 0; d < 3; ++d)
                  plane_n[j][d] = fitted[j].n[d] / fitted[j].gradient;
                plane_d[j] = fitted[j].d / fitted[j].gradient;
                grad_sum += fitted[j].gradient;
                ++grad_n;
              }
              gradient = grad_sum / grad_n;
              for (int i = 0; i < n_pool; ++i) {
                unsigned char best = k_no_plane;
                double best_d = k_crossing_band * h;
                for (int j = 0; j < k_plane_cap; ++j) {
                  if (!plane_ok[j])
                    continue;
                  const double d =
                      std::abs(plane_n[j][0] * pool_p[i][0] +
                               plane_n[j][1] * pool_p[i][1] +
                               plane_n[j][2] * pool_p[i][2] + plane_d[j]);
                  if (d < best_d) {
                    best_d = d;
                    best = static_cast<unsigned char>(j);
                  }
                }
                assign[i] = best;
              }
            }

            // rebuild each own component's QEF from the fitted planes; a
            // crossing on none of them keeps its own estimated plane
            for (int comp = 0; comp < own_n_comp; ++comp) {
              int comp_begin = own_comp_begin[comp];
              int comp_end = own_comp_end[comp];
              if (comp_end - comp_begin < 2)
                continue;
              tf::remesh::quadric q{};
              double mass[3] = {0, 0, 0};
              int n_own = 0;
              int n_used_planes = 0;
              unsigned used = 0;
              for (int i = comp_begin; i < comp_end; ++i) {
                double n[3], d_plane;
                int j = assign[i];
                if (j != k_no_plane && plane_ok[j]) {
                  if (!((used >> j) & 1u)) {
                    used |= 1u << j;
                    ++n_used_planes;
                  }
                  n[0] = plane_n[j][0];
                  n[1] = plane_n[j][1];
                  n[2] = plane_n[j][2];
                  d_plane = plane_d[j];
                } else {
                  n[0] = pool_n[i][0];
                  n[1] = pool_n[i][1];
                  n[2] = pool_n[i][2];
                  d_plane = -(n[0] * pool_p[i][0] + n[1] * pool_p[i][1] +
                              n[2] * pool_p[i][2]);
                }
                int ai = 0;
                for (int a = 0; a < 3; ++a)
                  for (int b = a; b < 3; ++b, ++ai)
                    q.A[ai] += n[a] * n[b];
                q.b[0] += d_plane * n[0];
                q.b[1] += d_plane * n[1];
                q.b[2] += d_plane * n[2];
                q.c += d_plane * d_plane;
                for (int d = 0; d < 3; ++d)
                  mass[d] += pool_p[i][d];
                ++n_own;
              }
              if (n_own == 0)
                continue; // nothing of this component's own is in the pool
              auto center =
                  tf::make_point(Real(mass[0] / n_own), Real(mass[1] / n_own),
                                 Real(mass[2] / n_own));
              // The stabilizer is insurance against an ESTIMATED plane's
              // ill-conditioning, and it is paid by exactly the shallow
              // features that can least afford it. A certified fit is the
              // field's own plane, so it pays the premium only in the
              // direction the quadric leaves free — along a crease, where the
              // crossing centroid is the answer.
              const double trace = (q.A[0] + q.A[3] + q.A[5]) / 3.0;
              double free_lambda = k_certified_stabilizer * trace;
              if (!certified) {
                const double lambda = stabilizer * trace;
                q.A[0] += lambda;
                q.A[3] += lambda;
                q.A[5] += lambda;
                for (int d = 0; d < 3; ++d)
                  q.b[d] -= lambda * double(center[d]);
                free_lambda = 0;
              }
              auto vertex = center;
              auto state = refit_state::clamped;
              // A component whose crossings lie on fewer than two of the
              // certified planes lies on no feature those planes state: a
              // fillet's arc is exactly this, its two flanks certified and its
              // own surface on neither.
              if (certified && n_used_planes < 2)
                state = refit_state::no_feature;
              else if (auto opt = tf::remesh::solve_optimal_quadric<Real>(
                           q, center, free_lambda)) {
                const double solved[3] = {double((*opt)[0]), double((*opt)[1]),
                                          double((*opt)[2])};
                // a certified solve is allowed further out: the apex of a
                // shallow wedge or cone genuinely sits there
                const int margin = certified ? k_certified_escape : 1;
                bool contained = true;
                for (int d = 0; d < 3; ++d) {
                  const int cell = d == 0 ? x : (d == 1 ? y : z);
                  const double lo =
                      cells.g.origin[d] + (cell - margin) * cells.g.spacing[d];
                  contained =
                      contained && solved[d] >= lo &&
                      solved[d] <= lo + (2 * margin + 1) * cells.g.spacing[d];
                }
                if (!contained)
                  state = refit_state::clamped;
                else if (certified &&
                         !tangency_admits(sample_p, sample_v, n_samples, solved,
                                          gradient, k_tangency_band * h))
                  state = refit_state::off_surface;
                else {
                  vertex = *opt;
                  state = refit_state::refit;
                }
              }
              // the same ownership the vertex pass used: a component's ids are
              // its intervals, not its index
              const int id_base =
                  component_id_base(own_case, own_retained, comp);
              const int id_copies =
                  component_id_copies(own_case, own_retained, comp);
              // only a refit states a vertex: every other verdict — the samples
              // refusing it, the planes stating no feature, a solve that left
              // its cells or did not solve at all — keeps the one the vertex
              // pass gave it, which is its own cell's answer rather than this
              // neighbourhood's centroid
              const bool keep = state != refit_state::refit;
              for (int c = 0; c < id_copies; ++c) {
                const auto id = own_base + Index(id_base + c);
                if constexpr (Record) {
                  const double step[3] = {
                      double(vertex[0]) - double(points[id][0]),
                      double(vertex[1]) - double(points[id][1]),
                      double(vertex[2]) - double(points[id][2])};
                  provenance[id] = refit_record{
                      state, static_cast<unsigned char>(n_used_planes),
                      certified,
                      keep ? 0.f
                           : float(std::sqrt(step[0] * step[0] +
                                             step[1] * step[1] +
                                             step[2] * step[2]))};
                }
                if (keep)
                  continue;
                points[id][0] = vertex[0];
                points[id][1] = vertex[1];
                points[id][2] = vertex[2];
              }
            } // component
          }
        });
  }
}

} // namespace volume_detail
} // namespace tf

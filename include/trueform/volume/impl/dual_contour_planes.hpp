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
#include "../../core/point.hpp"
#include "../../core/vector.hpp"
#include "../../core/external/miniselect/pdqselect.h"
#include "./dual_contour_cells.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>

namespace tf {
namespace volume_detail {

// The 4x4x4 samples whose cells are a flagged cell's 3x3x3 pool.
constexpr int k_pool_sample_cap = 64;

// Planes one pool can state at once.
constexpr int k_plane_cap = 3;

// The band a seeded plane admits samples within, in cells: the clustered
// planes are only close, so the first round has to reach past their error.
inline constexpr double k_seed_band = 0.4;

// After that a plane admits what it explains to within this multiple of the
// median deviation of the samples it holds, never below the floor. A band that
// shrank on a schedule instead would lock the seed's error in, because the
// samples that would correct a tilted plane are exactly the far ones a tight
// band throws away.
inline constexpr double k_band_scale = 4.0;
inline constexpr double k_band_floor = 0.01;
inline constexpr int k_regression_rounds = 5;

// A plane whose own samples lie this close to it, in cells, is the region's
// exact plane: a distance field is affine in a face's nearest-feature region
// and curved everywhere else, so nothing else fits this well.
inline constexpr double k_plane_certificate = 0.006;

// Fewer samples than this state a plane by accident.
inline constexpr int k_plane_support = 6;

// The reject band a crossing is held to when certified planes decide which of
// them its component's quadric reads.
inline constexpr double k_crossing_band = 0.5;

// A sample or crossing that lies on no plane.
constexpr unsigned char k_no_plane = 0xff;

/// @brief One plane of a flagged cell's neighbourhood, with the certificate its
/// own samples give it.
///
/// The model is the field itself: `n . p + d` is the value the plane's region
/// carries, so @ref n is that region's gradient and is unit for a distance
/// field. @ref residual is the RMS deviation of that model over the samples the
/// plane kept and @ref spread their median, both as world lengths.
///
/// The length of that gradient turns a value into a length, and every reader of
/// the plane needs it. It is a fact of the normal, so @ref state is the one
/// place either is written and @ref gradient is not derived twice.
struct pool_plane {
  double n[3] = {0, 0, 0};
  double d = 0;
  double gradient = 0; ///< `|n|`, stated with the normal it belongs to
  double residual = 0;
  double spread = 0;
  int support = 0;
  bool ok = false;
  bool certified = false;

  auto state(double n0, double n1, double n2, double offset) -> void {
    n[0] = n0;
    n[1] = n1;
    n[2] = n2;
    d = offset;
    gradient = std::sqrt(n0 * n0 + n1 * n1 + n2 * n2);
  }
};

/// @brief How far a sample sits from what a plane says its value should be.
inline auto plane_deviation(const pool_plane &plane,
                            const tf::point<double, 3> &p, double v) -> double {
  if (plane.gradient < 1e-12)
    return 1e30;
  return std::abs(plane.n[0] * p[0] + plane.n[1] * p[1] + plane.n[2] * p[2] +
                  plane.d - v) /
         plane.gradient;
}

/// @brief Whether two fitted planes are the same plane. A cluster split across
/// one face states it twice, which wastes the slot a missing face needs.
inline auto same_pool_plane(const pool_plane &a, const pool_plane &b, double h)
    -> bool {
  if (a.gradient < 1e-12 || b.gradient < 1e-12)
    return false;
  const double dot = (a.n[0] * b.n[0] + a.n[1] * b.n[1] + a.n[2] * b.n[2]) /
                     (a.gradient * b.gradient);
  return dot > 0.99 &&
         std::abs(a.d / a.gradient - b.d / b.gradient) < k_seed_band * h;
}

/// @brief The samples of a flagged cell's pool: the 4x4x4 block whose cells are
/// its 3x3x3 neighbourhood, clipped to the grid.
template <typename Compute, typename Samples>
inline auto gather_pool_samples(const dc_grid &g, const Samples &samples,
                                Compute iso, int x, int y, int z,
                                tf::point<double, 3> *p, double *v) -> int {
  int n = 0;
  for (int k = 0; k < 4; ++k) {
    const int sz = z - 1 + k;
    if (sz < 0 || sz >= g.nz)
      continue;
    for (int j = 0; j < 4; ++j) {
      const int sy = y - 1 + j;
      if (sy < 0 || sy >= g.ny)
        continue;
      for (int i = 0; i < 4; ++i) {
        const int sx = x - 1 + i;
        if (sx < 0 || sx >= g.nx)
          continue;
        p[n] = tf::make_point(g.origin[0] + sx * g.spacing[0],
                              g.origin[1] + sy * g.spacing[1],
                              g.origin[2] + sz * g.spacing[2]);
        v[n] = double(
            static_cast<Compute>(samples[g.sample_index(sx, sy, sz)]) - iso);
        ++n;
      }
    }
  }
  return n;
}

/// @brief Least squares of the sample values on the sample positions, over the
/// samples @p ids names.
///
/// The normal equations are formed about the subset's own mean, which leaves a
/// 3x3 symmetric system in the gradient alone; the offset follows. Positions
/// that span no volume state no plane, and neither do too few of them.
inline auto least_squares_plane(const tf::point<double, 3> *p, const double *v,
                                const int *ids, int m, pool_plane &out)
    -> bool {
  if (m < k_plane_support)
    return false;
  double mp[3] = {0, 0, 0}, mv = 0;
  for (int k = 0; k < m; ++k) {
    const int i = ids[k];
    for (int d = 0; d < 3; ++d)
      mp[d] += p[i][d];
    mv += v[i];
  }
  for (auto &c : mp)
    c /= m;
  mv /= m;
  double c00 = 0, c01 = 0, c02 = 0, c11 = 0, c12 = 0, c22 = 0;
  double b[3] = {0, 0, 0};
  for (int k = 0; k < m; ++k) {
    const int i = ids[k];
    const double q[3] = {p[i][0] - mp[0], p[i][1] - mp[1], p[i][2] - mp[2]};
    const double w = v[i] - mv;
    c00 += q[0] * q[0];
    c01 += q[0] * q[1];
    c02 += q[0] * q[2];
    c11 += q[1] * q[1];
    c12 += q[1] * q[2];
    c22 += q[2] * q[2];
    for (int d = 0; d < 3; ++d)
      b[d] += q[d] * w;
  }
  const double det = c00 * (c11 * c22 - c12 * c12) -
                     c01 * (c01 * c22 - c12 * c02) +
                     c02 * (c01 * c12 - c11 * c02);
  const double frob2 = c00 * c00 + c11 * c11 + c22 * c22 +
                       2.0 * (c01 * c01 + c02 * c02 + c12 * c12);
  if (!(std::abs(det) > 1e-9 * frob2 * std::sqrt(frob2)))
    return false;
  const double inv = 1.0 / det;
  const double a[3] = {
      inv * ((c11 * c22 - c12 * c12) * b[0] + (c02 * c12 - c01 * c22) * b[1] +
             (c01 * c12 - c02 * c11) * b[2]),
      inv * ((c02 * c12 - c01 * c22) * b[0] + (c00 * c22 - c02 * c02) * b[1] +
             (c01 * c02 - c00 * c12) * b[2]),
      inv * ((c01 * c12 - c02 * c11) * b[0] + (c01 * c02 - c00 * c12) * b[1] +
             (c00 * c11 - c01 * c01) * b[2])};
  if (a[0] * a[0] + a[1] * a[1] + a[2] * a[2] < 1e-24)
    return false;
  out.state(a[0], a[1], a[2], mv - (a[0] * mp[0] + a[1] * mp[1] + a[2] * mp[2]));
  out.support = m;
  return true;
}

/// @brief The plane a labelled subset of the samples states, and how well.
///
/// Least squares has no defence against what a wide band lets in — a handful of
/// samples from the curved region beside a crease tilt it enough that every
/// sample then looks equally wrong, which is a state the fit never leaves. So
/// it is taken twice: once over the whole labelled subset, and again over the
/// half of it the first fit explains best. The region's own samples sit at
/// zero, so that half is them, and the second fit is the region's own plane.
inline auto fit_sample_plane(const tf::point<double, 3> *p, const double *v,
                             const unsigned char *label, int n_samples,
                             unsigned char want, double floor_band,
                             pool_plane &out) -> void {
  int own[k_pool_sample_cap];
  int n_own = 0;
  for (int i = 0; i < n_samples; ++i)
    if (label[i] == want)
      own[n_own++] = i;
  pool_plane first{};
  if (!least_squares_plane(p, v, own, n_own, first)) {
    out.ok = false;
    out.certified = false;
    out.support = 0;
    return;
  }
  // how far the first fit leaves each of its own samples: the median of those
  // is the band, and the half within it is the subset the second fit reads.
  // Only the value at the middle rank is read, and the samples it selects are
  // taken from the unpermuted copy, so the selector owes no order beyond it.
  double own_dev[k_pool_sample_cap];
  double dev[k_pool_sample_cap];
  for (int k = 0; k < n_own; ++k)
    dev[k] = own_dev[k] = plane_deviation(first, p[own[k]], v[own[k]]);
  miniselect::pdqselect_branchless(dev, dev + n_own / 2, dev + n_own);
  const double half = std::max(dev[n_own / 2], floor_band);
  int kept[k_pool_sample_cap];
  int n_kept = 0;
  for (int k = 0; k < n_own; ++k)
    if (own_dev[k] <= half)
      kept[n_kept++] = own[k];
  pool_plane second{};
  const bool concentrated = least_squares_plane(p, v, kept, n_kept, second);
  if (concentrated)
    out.state(second.n[0], second.n[1], second.n[2], second.d);
  else
    out.state(first.n[0], first.n[1], first.n[2], first.d);
  out.support = concentrated ? second.support : first.support;
  const int *held = concentrated ? kept : own;
  const int n_held = concentrated ? n_kept : n_own;
  double sq = 0;
  for (int k = 0; k < n_held; ++k) {
    const double r = plane_deviation(out, p[held[k]], v[held[k]]);
    sq += r * r;
    dev[k] = r;
  }
  out.residual = std::sqrt(sq / n_held);
  miniselect::pdqselect_branchless(dev, dev + n_held / 2, dev + n_held);
  out.spread = dev[n_held / 2];
  out.ok = true;
}

/// @brief Label every sample with the plane that explains it best, and refit
/// each plane to what it took.
///
/// A plane admits a sample within the band its own spread states, so the
/// labelling and the fitting chase each other down: a plane that is only close
/// reaches widely, and one that is exact takes its region and nothing else.
///
/// A plane whose labelled subset came out of a round exactly as it went in
/// would refit to what it already is, so it is not refitted; when no plane's
/// subset moved, neither will any later round's, and the chase is over.
inline auto run_regression_rounds(const tf::point<double, 3> *p,
                                  const double *v, int n_samples, double h,
                                  int rounds, pool_plane *planes,
                                  unsigned char *label) -> void {
  unsigned char settled[k_pool_sample_cap];
  for (int round = 0; round < rounds; ++round) {
    double band[k_plane_cap];
    for (int j = 0; j < k_plane_cap; ++j)
      band[j] = std::min(k_seed_band * h,
                         std::max(k_band_floor * h,
                                  k_band_scale * planes[j].spread));
    for (int i = 0; i < n_samples; ++i) {
      unsigned char best = k_no_plane;
      double best_r = k_seed_band * h;
      for (int j = 0; j < k_plane_cap; ++j) {
        if (!planes[j].ok)
          continue;
        const double r = plane_deviation(planes[j], p[i], v[i]);
        if (r < band[j] && r < best_r) {
          best_r = r;
          best = static_cast<unsigned char>(j);
        }
      }
      label[i] = best;
    }
    unsigned moved = round == 0 ? ~0u : 0u;
    for (int i = 0; i < n_samples && round > 0; ++i) {
      if (label[i] == settled[i])
        continue;
      if (label[i] != k_no_plane)
        moved |= 1u << label[i];
      if (settled[i] != k_no_plane)
        moved |= 1u << settled[i];
    }
    if (moved == 0u)
      return;
    std::memcpy(settled, label, std::size_t(n_samples));
    for (int j = 0; j < k_plane_cap; ++j)
      if (planes[j].ok && ((moved >> j) & 1u))
        fit_sample_plane(p, v, label, n_samples, static_cast<unsigned char>(j),
                         k_band_floor * h, planes[j]);
  }
}

/// @brief Whether a plane certifies: enough samples of its own, lying on it.
inline auto certify_pool_plane(pool_plane &plane, double h) -> bool {
  plane.certified = plane.ok && plane.support >= k_plane_support &&
                    plane.residual <= k_plane_certificate * h;
  return plane.certified;
}

/// @brief Retire every plane a lower slot already states.
///
/// A cluster split across one face states that face twice, and a crossing
/// handed to the copy is a crossing whose own face never entered the quadric —
/// while the copy also occupies the slot a face the clusters missed needs. The
/// one producer of that fact; every regression is followed by it.
///
/// @return How many planes it retired.
inline auto retire_duplicate_planes(pool_plane *planes, double h) -> int {
  int retired = 0;
  for (int j = 0; j < k_plane_cap; ++j)
    for (int i = 0; i < j; ++i)
      if (planes[i].certified && planes[j].certified &&
          same_pool_plane(planes[i], planes[j], h)) {
        planes[j].certified = false;
        planes[j].ok = false;
        ++retired;
      }
  return retired;
}

/// @brief Refit the seeded planes to the samples they explain, and certify each
/// against its own.
///
/// A distance field is EXACTLY `n . p + d` inside a face's nearest-feature
/// region, so four samples of that region determine the plane outright. Which
/// samples those are is the whole question, and the model answers it itself: a
/// sample the plane does not explain — the curved region outside a convex
/// crease, another feature's region — excludes itself through its own
/// deviation.
///
/// @param seed_n Unit normals of the seeded planes, @p seed_d their offsets,
///        @p seed_ok which of the @ref k_plane_cap slots were seeded at all.
///        Read only; the seeds stay the caller's.
/// @return How many planes came out certified, each of them distinct.
inline auto regress_pool_planes(const tf::point<double, 3> *p, const double *v,
                                int n_samples, double h,
                                const double (*seed_n)[3], const double *seed_d,
                                const bool *seed_ok, pool_plane *planes)
    -> int {
  for (int j = 0; j < k_plane_cap; ++j) {
    planes[j] = pool_plane{};
    planes[j].ok = seed_ok[j];
    if (!planes[j].ok)
      continue;
    planes[j].state(seed_n[j][0], seed_n[j][1], seed_n[j][2], seed_d[j]);
    planes[j].spread = k_seed_band * h;
  }
  unsigned char label[k_pool_sample_cap];
  run_regression_rounds(p, v, n_samples, h, k_regression_rounds, planes, label);
  int certified = 0;
  for (int j = 0; j < k_plane_cap; ++j)
    certified += certify_pool_plane(planes[j], h);
  return certified - retire_duplicate_planes(planes, h);
}

/// @brief A slot no certified plane occupies, or -1.
inline auto free_plane_slot(const pool_plane *planes) -> int {
  int slot = -1;
  for (int j = 0; j < k_plane_cap; ++j)
    if (!planes[j].certified)
      slot = j;
  return slot;
}

/// @brief The planes a flagged cell's neighbourhood lies on.
///
/// The clustered crossings seed it and the samples state it. Where three faces
/// meet, a cluster of crossing NORMALS is a poor seed: the blended normals
/// between the faces are what a farthest-point seed reaches first, so one face
/// can go unstated and the corner comes out a crease. The crossings themselves
/// answer that — one of them lies on the missing face and carries its own
/// estimate of the normal — so a crossing no certified plane explains seeds the
/// free slot, and the samples are asked again.
///
/// The seeds are the CALLER's planes and stay its own: the second round's are
/// this function's scratch, so a call that certifies too few leaves the caller
/// holding exactly the planes its own clustering assigned its crossings to.
inline auto state_pool_planes(const tf::point<double, 3> *sample_p,
                              const double *sample_v, int n_samples,
                              const tf::point<double, 3> *pool_p,
                              const tf::vector<double, 3> *pool_n, int n_pool,
                              double h, const double (*seed_n)[3],
                              const double *seed_d, const bool *seed_ok,
                              pool_plane *planes) -> int {
  const int certified = regress_pool_planes(sample_p, sample_v, n_samples, h,
                                            seed_n, seed_d, seed_ok, planes);
  if (certified == 0)
    return certified;
  const int slot = free_plane_slot(planes);
  if (slot < 0)
    return certified;
  int unexplained = -1;
  double farthest = k_crossing_band * h;
  for (int i = 0; i < n_pool; ++i) {
    if (pool_n[i][0] == 0.0 && pool_n[i][1] == 0.0 && pool_n[i][2] == 0.0)
      continue;
    double nearest = 1e30;
    for (int j = 0; j < k_plane_cap; ++j)
      if (planes[j].certified)
        nearest = std::min(nearest, plane_deviation(planes[j], pool_p[i], 0.0));
    if (nearest > farthest) {
      farthest = nearest;
      unexplained = i;
    }
  }
  if (unexplained < 0)
    return certified;

  double next_n[k_plane_cap][3]{};
  double next_d[k_plane_cap]{};
  bool next_ok[k_plane_cap]{};
  for (int j = 0; j < k_plane_cap; ++j) {
    next_ok[j] = planes[j].certified;
    if (!next_ok[j])
      continue;
    for (int d = 0; d < 3; ++d)
      next_n[j][d] = planes[j].n[d] / planes[j].gradient;
    next_d[j] = planes[j].d / planes[j].gradient;
  }
  next_ok[slot] = true;
  for (int d = 0; d < 3; ++d)
    next_n[slot][d] = pool_n[unexplained][d];
  next_d[slot] = -(next_n[slot][0] * pool_p[unexplained][0] +
                   next_n[slot][1] * pool_p[unexplained][1] +
                   next_n[slot][2] * pool_p[unexplained][2]);
  return regress_pool_planes(sample_p, sample_v, n_samples, h, next_n, next_d,
                             next_ok, planes);
}

/// @brief Whether the samples admit a surface point at @p x.
///
/// A sample's value is the distance to the NEAREST surface point, so no surface
/// point is closer to it than that. A vertex that breaks the inequality is not
/// on the surface the samples state: the phantom corner where a fillet's two
/// flanks meet is exactly such a point, and so is the meet of two walls that
/// belong to different features.
///
/// @param gradient The fitted gradient magnitude, which turns a field value
///        into a length.
inline auto tangency_admits(const tf::point<double, 3> *p, const double *v,
                            int n_samples, const double *x, double gradient,
                            double tolerance) -> bool {
  const double inv = gradient > 1e-12 ? 1.0 / gradient : 0.0;
  for (int i = 0; i < n_samples; ++i) {
    const double q[3] = {x[0] - p[i][0], x[1] - p[i][1], x[2] - p[i][2]};
    const double reach =
        std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]) + tolerance;
    if (std::abs(v[i]) * inv > reach)
      return false;
  }
  return true;
}

} // namespace volume_detail
} // namespace tf

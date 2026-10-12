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

#include "../../core/point.hpp"
#include "../canonical_plane.hpp"
#include "../meta.hpp"
#include "./admits_placement.hpp"
#include "./place_at_exact_meet.hpp"
#include "./place_on_exact_line.hpp"
#include "./pool/certificate_census.hpp"
#include "./pool/certificate_lane.hpp"
#include "./pool/certify_exact.hpp"
#include "./pool/exact_lane.hpp"
#include "./pool/wide_to_double.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace tf::exact::door {

/// The one-plane solve a block carries from vertex to vertex: the frames
/// of the last plane it landed on and the step its last residual bought.
/// A pooled wall's vertices are consecutive in their own mesh, so one
/// reduction answers a whole wall; a different plane resets it.
template <typename Int> struct plane_landing {
  tf::exact::canonical_plane<Int> plane{};
  pool::certificate_lane<Int> lane{};
  pool::solved_step<Int> step{};
  pool::certificate_census census{};
  bool held = false;
};

/// Where one pulled vertex stands, and whether it stands on every plane it
/// owes. `placed` with the input point is a vertex already on all of them;
/// not `placed` is a refusal, and the point is the input's.
template <typename Int> struct vertex_placement {
  tf::point<Int, 3> point{};
  bool placed = false;
};

/// Where one pulled vertex stands: a lattice point within the band ON
/// every committed plane, or the vertex where the input put it.
///
/// `committed` are the planes the vertex must satisfy exactly, sorted and
/// unique; `helpers` are the vertex's own uncommitted faces' planes, which
/// may choose WHERE on the committed planes it lands but are never owed.
/// A vertex already on every committed plane does not move. Otherwise the
/// meet of the widest triple holding a committed plane, then the line of
/// the widest such pair, then the committed planes alone — every triple,
/// every pair, each plane — and the first answer on all of them within the
/// band stands. When none does the vertex stays: the band never invents a
/// position.
template <typename Int, typename Committed, typename Helpers>
auto place_vertex(const tf::point<Int, 3> &original, const Committed &committed,
                  const Helpers &helpers, Int tolerance,
                  plane_landing<Int> &landing) -> vertex_placement<Int> {
  using T1 = typename tf::exact::meta<Int>::T1;
  using product_type = typename pool::exact_lane<Int>::product_type;

  const auto on_all = [&committed](const tf::point<Int, 3> &at) {
    for (const auto &plane : committed)
      if (tf::exact::orient3d_plane_sign<Int>(plane, at) != 0)
        return false;
    return true;
  };
  const auto stands = [&](const tf::point<Int, 3> &at) {
    return on_all(at) &&
           admits_placement<Int>(
               original, std::array<T1, 3>{T1(at[0]), T1(at[1]), T1(at[2])},
               tolerance);
  };
  if (on_all(original))
    return {original, true};

  const std::size_t n_owed = committed.size();
  const std::size_t n = n_owed + helpers.size();
  const auto plane_at = [&](std::size_t i) -> const auto & {
    return i < n_owed ? committed[i] : helpers[i - n_owed];
  };
  const auto normal_of = [&](std::size_t i) {
    const auto &plane = plane_at(i);
    return std::array<typename pool::exact_lane<Int>::coefficient_type, 3>{
        plane[0], plane[1], plane[2]};
  };
  // A plane or a line farther than the band from the vertex holds no point
  // the band admits, so no solve on it can stand. The distances are taken in
  // double and judged with a margin above their error, and a line of planes
  // too near parallel for that is left to the exact solve.
  const double band = double(tolerance);
  const auto residual = [&](std::size_t i) {
    const auto &plane = plane_at(i);
    const double n0 = pool::wide_to_double(plane[0]),
                 n1 = pool::wide_to_double(plane[1]),
                 n2 = pool::wide_to_double(plane[2]);
    return pool::wide_to_double(
               tf::exact::orient3d_plane_value<Int>(plane, original)) /
           std::sqrt(n0 * n0 + n1 * n1 + n2 * n2);
  };
  const auto line_beyond_band = [&](std::size_t a, std::size_t b) {
    const auto u = normal_of(a), v = normal_of(b);
    double cross2 = 0.0, uu = 0.0, vv = 0.0;
    for (std::size_t k = 0; k < 3; ++k) {
      const std::size_t k1 = (k + 1) % 3, k2 = (k + 2) % 3;
      const double c =
          pool::wide_to_double(product_type(u[k1]) * product_type(v[k2]) -
                               product_type(u[k2]) * product_type(v[k1]));
      cross2 += c * c;
      uu += pool::wide_to_double(u[k]) * pool::wide_to_double(u[k]);
      vv += pool::wide_to_double(v[k]) * pool::wide_to_double(v[k]);
    }
    const double sine2 = cross2 / (uu * vv);
    if (!(sine2 > 1e-6))
      return false;
    const double ra = residual(a), rb = residual(b);
    double cosine = 0.0;
    for (std::size_t k = 0; k < 3; ++k)
      cosine += pool::wide_to_double(u[k]) * pool::wide_to_double(v[k]);
    cosine /= std::sqrt(uu * vv);
    const double distance2 =
        ((ra - rb) * (ra - rb) + 2.0 * (1.0 - cosine) * ra * rb) / sine2;
    return distance2 > band * band * (1.0 + 1e-6);
  };
  for (std::size_t a = 0; a < n_owed; ++a)
    if (std::fabs(residual(a)) > band * (1.0 + 1e-9))
      return {original, false};
  for (std::size_t a = 0; a < n_owed; ++a)
    for (std::size_t b = a + 1; b < n_owed; ++b)
      if (line_beyond_band(a, b))
        return {original, false};
  tf::point<Int, 3> at{};

  {
    product_type widest(0);
    std::size_t i0 = 0, i1 = 0, i2 = 0;
    for (std::size_t a = 0; a < n_owed; ++a)
      for (std::size_t b = a + 1; b < n; ++b)
        for (std::size_t c = b + 1; c < n; ++c) {
          product_type measure =
              exact_det3<Int>(normal_of(a), normal_of(b), normal_of(c));
          if (measure < product_type(0))
            measure = -measure;
          if (measure > widest) {
            widest = measure;
            i0 = a;
            i1 = b;
            i2 = c;
          }
        }
    if (widest > product_type(0) &&
        place_at_exact_meet<Int>(plane_at(i0), plane_at(i1), plane_at(i2),
                                 at) &&
        stands(at))
      return {at, true};
  }
  {
    double widest = 0.0;
    std::size_t j0 = 0, j1 = 0;
    for (std::size_t a = 0; a < n_owed; ++a)
      for (std::size_t b = a + 1; b < n; ++b) {
        const auto u = normal_of(a), v = normal_of(b);
        double measure = 0.0;
        for (std::size_t k = 0; k < 3; ++k) {
          const std::size_t k1 = (k + 1) % 3, k2 = (k + 2) % 3;
          const double c =
              pool::wide_to_double(product_type(u[k1]) * product_type(v[k2]) -
                                   product_type(u[k2]) * product_type(v[k1]));
          measure += c * c;
        }
        if (measure > widest) {
          widest = measure;
          j0 = a;
          j1 = b;
        }
      }
    if (widest > 0.0 && !line_beyond_band(j0, j1) &&
        place_on_exact_line<Int>(plane_at(j0), plane_at(j1), original, at) &&
        stands(at))
      return {at, true};
  }
  for (std::size_t a = 0; a < n_owed; ++a)
    for (std::size_t b = a + 1; b < n_owed; ++b)
      for (std::size_t c = b + 1; c < n_owed; ++c)
        if (place_at_exact_meet<Int>(committed[a], committed[b], committed[c],
                                     at) &&
            stands(at))
          return {at, true};
  for (std::size_t a = 0; a < n_owed; ++a)
    for (std::size_t b = a + 1; b < n_owed; ++b)
      if (place_on_exact_line<Int>(committed[a], committed[b], original, at) &&
          stands(at))
        return {at, true};
  for (const auto &plane : committed) {
    if (!landing.held || landing.plane != plane) {
      landing.plane = plane;
      landing.lane = pool::certificate_lane<Int>{};
      landing.step = pool::solved_step<Int>{};
      landing.held = true;
    }
    const auto certificate = pool::certify_exact<Int>(
        plane, original, tolerance, landing.lane, landing.step, landing.census);
    if (certificate.certified && on_all(certificate.point))
      return {certificate.point, true};
  }
  return {original, false};
}

} // namespace tf::exact::door

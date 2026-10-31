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
#include "../../core/dot.hpp"
#include "../../core/epsilon.hpp"
#include "../../core/point.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tf::geometry {

/// The least shape, twice the area over the longest edge squared, a face
/// unfolds at. The virtual source is the meet of two circles whose radii carry
/// the revision tolerance, and a needle amplifies that by its aspect, so below
/// this shape the unfolding is no longer a statement and the edge is. The
/// value is the largest that leaves the reference mesh's every bit unchanged.
inline constexpr double geodesic_needle_shape = 1.5e-4;

/// The distance at `i` that a front known at `j` and `k` states across the
/// triangle `(i, j, k)`, read from the triangle's lengths `l_ij`, `l_ik`,
/// `l_jk` and twice its area, `two_area`.
///
/// The front is unfolded into the triangle's plane: the virtual source stands
/// at `d_j` from `j` and `d_k` from `k`, on the far side of the edge `jk`,
/// and `i` reads its distance to it. Where the two circles do not meet, or
/// the source falls outside the angle the triangle has at `i`, the front
/// crosses an edge instead, and the nearer of `d_j + l_ij` and `d_k + l_ik`
/// is the answer. A face below @ref tf::geometry::geodesic_needle_shape
/// unfolds nothing and the edge answers.
inline auto unfolded_distance(double l_ij, double l_ik, double l_jk,
                              double two_area, double d_j, double d_k)
    -> double {
  const double norm2_j = l_ij * l_ij;
  const double norm2_k = l_ik * l_ik;
  const double norm2_kj = l_jk * l_jk;
  const double inner = 0.5 * (norm2_j + norm2_k - norm2_kj);
  const double crossed = std::min(d_j + l_ij, d_k + l_ik);
  const double longest2 = std::max(norm2_j, std::max(norm2_k, norm2_kj));
  if (!(two_area >= tf::geometry::geodesic_needle_shape * longest2))
    return crossed;

  const double spread = (d_j - d_k) * (d_j - d_k);
  const double reach = (d_j + d_k) * (d_j + d_k);
  const double discriminant = (norm2_kj - spread) * (reach - norm2_kj);
  // Both degeneracy tests read ratios, so a fine or skinny triangle answers
  // as its scaled copy does.
  if (discriminant <= tf::epsilon<double> * norm2_kj * reach)
    return crossed;
  const double height = std::sqrt(discriminant);
  const double denominator = 2.0 * two_area * norm2_kj;
  const double x_j = (two_area * (norm2_kj + d_k * d_k - d_j * d_j) +
                      (norm2_k - inner) * height) /
                     denominator;
  const double x_k = (two_area * (norm2_kj + d_j * d_j - d_k * d_k) -
                      (inner - norm2_j) * height) /
                     denominator;
  if (!(x_j >= tf::epsilon<double>) || !(x_k >= tf::epsilon<double>))
    return crossed;
  return std::sqrt(std::max(0.0, x_j * x_j * norm2_j + x_k * x_k * norm2_k +
                                     2.0 * x_j * x_k * inner));
}

/// The same distance, read from the triangle's corners.
template <std::size_t Dims>
auto unfolded_distance(const tf::point<double, Dims> &i,
                       const tf::point<double, Dims> &j,
                       const tf::point<double, Dims> &k, double d_j, double d_k)
    -> double {
  const auto e_j = j - i;
  const auto e_k = k - i;
  const double inner = tf::dot(e_j, e_k);
  return tf::geometry::unfolded_distance(
      e_j.length(), e_k.length(), (k - j).length(),
      std::sqrt(std::max(0.0, e_j.length2() * e_k.length2() - inner * inner)),
      d_j, d_k);
}

} // namespace tf::geometry

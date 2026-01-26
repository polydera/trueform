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
#include "./sqrt.hpp"
#include <array>
namespace tf::core {
template <typename T>
auto eigen_values_of(const std::array<std::array<T, 3>, 3> &m) {
  using std::atan2;
  using std::cos;
  using std::sin;

  constexpr T inv3 = T(1) / T(3);
  const T sqrt3 = tf::sqrt(T(3));

  // Characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0
  // Eigenvalues are roots, all real for symmetric matrix
  T c0 = m[0][0] * m[1][1] * m[2][2] + T(2) * m[1][0] * m[2][0] * m[2][1] -
         m[0][0] * m[2][1] * m[2][1] - m[1][1] * m[2][0] * m[2][0] -
         m[2][2] * m[1][0] * m[1][0];
  T c1 = m[0][0] * m[1][1] - m[1][0] * m[1][0] + m[0][0] * m[2][2] -
         m[2][0] * m[2][0] + m[1][1] * m[2][2] - m[2][1] * m[2][1];
  T c2 = m[0][0] + m[1][1] + m[2][2];

  // Construct parameters for solving in closed form
  T c2_over_3 = c2 * inv3;
  T a_over_3 = (c2 * c2_over_3 - c1) * inv3;
  a_over_3 = std::max(a_over_3, T(0));

  T half_b = T(0.5) * (c0 + c2_over_3 * (T(2) * c2_over_3 * c2_over_3 - c1));

  T q = a_over_3 * a_over_3 * a_over_3 - half_b * half_b;
  q = std::max(q, T(0));

  // Compute eigenvalues by solving for roots
  T rho = tf::sqrt(a_over_3);
  T theta = atan2(tf::sqrt(q), half_b) * inv3;
  T cos_theta = cos(theta);
  T sin_theta = sin(theta);

  // Roots are already sorted, since cos is monotonically decreasing on [0, pi]
  std::array<T, 3> eigenvalues;
  eigenvalues[0] = c2_over_3 - rho * (cos_theta + sqrt3 * sin_theta);
  eigenvalues[1] = c2_over_3 - rho * (cos_theta - sqrt3 * sin_theta);
  eigenvalues[2] = c2_over_3 + T(2) * rho * cos_theta;

  return eigenvalues;
}

template <typename T>
auto eigen_values_of(const std::array<std::array<T, 2>, 2> &m) {
  using std::max;

  // Assume symmetric matrix:
  // [ a  b ]
  // [ b  c ]
  const T a = m[0][0];
  const T b = m[0][1]; // assume m[0][1] == m[1][0]
  const T c = m[1][1];

  // Eigenvalues of 2x2 symmetric:
  // λ = (trace ± sqrt(trace^2 - 4 det)) / 2
  const T trace       = a + c;
  const T half_trace  = trace / T(2);
  const T det         = a * c - b * b;

  T disc_sq = half_trace * half_trace - det;
  disc_sq   = max(disc_sq, T(0)); // clamp for numerical safety

  const T disc = tf::sqrt(disc_sq);

  T lambda0 = half_trace - disc;
  T lambda1 = half_trace + disc;

  if (lambda0 > lambda1)
    std::swap(lambda0, lambda1);

  return std::array<T, 2>{lambda0, lambda1};
}

} // namespace tf::core

/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
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
} // namespace tf::core

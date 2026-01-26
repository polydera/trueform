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

#include "./eigen_of_symmetric.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace tf {

/// @ingroup core_algorithms
/// @brief Compute SVD of a 3x3 symmetric matrix.
///
/// For symmetric matrices, SVD is derived from eigendecomposition:
/// singular values are absolute eigenvalues, sorted descending.
///
/// @tparam T The scalar type.
/// @param mat The symmetric 3x3 matrix.
/// @return A tuple of (singular values, left singular vectors U, right singular vectors V).
template <typename T>
auto svd_of_symmetric(const std::array<std::array<T, 3>, 3> &mat) {
  using std::abs;

  // Eigen decomposition (eigenvalues ascending, eigenvectors as unit_vector)
  auto ev = tf::eigen_of_symmetric(mat);
  const auto &lambdas = ev.first; // std::array<T,3>
  const auto &Q = ev.second;      // std::array<tf::unit_vector<T,3>,3>

  // Singular values: absolute eigenvalues
  std::array<T, 3> s{
      abs(lambdas[0]),
      abs(lambdas[1]),
      abs(lambdas[2]),
  };

  // V = Q (right singular vectors)
  std::array<tf::unit_vector<T, 3>, 3> V = Q;

  // U = Q * diag(sign(λ)) (left singular vectors)
  std::array<tf::unit_vector<T, 3>, 3> U = Q;
  if (lambdas[0] < T(0)) {
    U[0] = -U[0];
  }
  if (lambdas[1] < T(0)) {
    U[1] = -U[1];
  }
  if (lambdas[2] < T(0)) {
    U[2] = -U[2];
  }

  // Sort by descending singular value: permute s, U, V together
  std::array<std::size_t, 3> idx{0, 1, 2};
  std::sort(idx.begin(), idx.end(),
            [&](std::size_t i, std::size_t j) { return s[i] > s[j]; });

  std::array<T, 3> s_sorted{};
  std::array<tf::unit_vector<T, 3>, 3> U_sorted{};
  std::array<tf::unit_vector<T, 3>, 3> V_sorted{};

  for (std::size_t k = 0; k < 3; ++k) {
    const auto i = idx[k];
    s_sorted[k] = s[i];
    U_sorted[k] = U[i];
    V_sorted[k] = V[i];
  }

  return std::make_tuple(s_sorted, U_sorted, V_sorted);
}

/// @ingroup core_algorithms
/// @brief Compute SVD of a 2x2 symmetric matrix.
/// @overload
template <typename T>
auto svd_of_symmetric(const std::array<std::array<T, 2>, 2> &mat) {
  using std::abs;

  auto ev = tf::eigen_of_symmetric(mat);
  const auto &lambdas = ev.first; // std::array<T,2>
  const auto &Q = ev.second;      // std::array<tf::unit_vector<T,2>,2>

  std::array<T, 2> s{
      abs(lambdas[0]),
      abs(lambdas[1]),
  };

  std::array<tf::unit_vector<T, 2>, 2> V = Q;
  std::array<tf::unit_vector<T, 2>, 2> U = Q;

  if (lambdas[0] < T(0)) {
    U[0] = -U[0];
  }
  if (lambdas[1] < T(0)) {
    U[1] = -U[1];
  }

  std::array<std::size_t, 2> idx{0, 1};
  std::sort(idx.begin(), idx.end(),
            [&](std::size_t i, std::size_t j) { return s[i] > s[j]; });

  std::array<T, 2> s_sorted{};
  std::array<tf::unit_vector<T, 2>, 2> U_sorted{};
  std::array<tf::unit_vector<T, 2>, 2> V_sorted{};

  for (std::size_t k = 0; k < 2; ++k) {
    const auto i = idx[k];
    s_sorted[k] = s[i];
    U_sorted[k] = U[i];
    V_sorted[k] = V[i];
  }

  return std::make_tuple(s_sorted, U_sorted, V_sorted);
}

} // namespace tf

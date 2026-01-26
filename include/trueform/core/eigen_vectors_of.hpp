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
#include "./normalized.hpp"
#include "./unit_vector.hpp"
#include "./vector.hpp"

namespace tf::core {
namespace impl {
template <typename T>
auto extract_kernel_3x3(const std::array<std::array<T, 3>, 3> &mat,
                        tf::vector<T, 3> &representative) {
  using std::abs;

  // Find non-zero column i0 (precompute abs values)
  T abs_diag[3] = {abs(mat[0][0]), abs(mat[1][1]), abs(mat[2][2])};

  std::size_t i0 = 0;
  if (abs_diag[1] > abs_diag[0])
    i0 = 1;
  if (abs_diag[2] > abs_diag[i0])
    i0 = 2;

  representative[0] = mat[0][i0];
  representative[1] = mat[1][i0];
  representative[2] = mat[2][i0];

  // Get the other two columns
  std::size_t j1 = (i0 + 1) % 3;
  std::size_t j2 = (i0 + 2) % 3;

  tf::vector<T, 3> col1{mat[0][j1], mat[1][j1], mat[2][j1]};
  tf::vector<T, 3> col2{mat[0][j2], mat[1][j2], mat[2][j2]};

  // Compute cross products
  auto c0 = tf::cross(representative, col1);
  auto c1 = tf::cross(representative, col2);

  T n0 = tf::dot(c0, c0);
  T n1 = tf::dot(c1, c1);

  return (n0 > n1) ? c0 / tf::sqrt(n0) : c1 / tf::sqrt(n1);
}
} // namespace impl

template <typename T>
auto eigen_vectors_of(const std::array<std::array<T, 3>, 3> &m,
                      const std::array<T, 3> &eigenvalues) {
  using std::abs;

  std::array<tf::vector<T, 3>, 3> eigenvectors;

  // Shift matrix to mean eigenvalue and scale to [-1:1]
  T shift = (m[0][0] + m[1][1] + m[2][2]) / T(3);

  std::array<std::array<T, 3>, 3> scaled_mat;

  // Apply shift
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      scaled_mat[i][j] = m[i][j];
    }
  }
  scaled_mat[0][0] -= shift;
  scaled_mat[1][1] -= shift;
  scaled_mat[2][2] -= shift;

  // Find max absolute value for scaling
  T scale = abs(scaled_mat[0][0]);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      scale = std::max(scale, abs(scaled_mat[i][j]));
    }
  }

  // Scale matrix and eigenvalues
  T inv_scale = (scale > T(0)) ? (T(1) / scale) : T(1);
  for (std::size_t i = 0; i < 3; ++i) {
    for (std::size_t j = 0; j < 3; ++j) {
      scaled_mat[i][j] *= inv_scale;
    }
  }

  tf::vector<T, 3> scaled_eigenvalues{(eigenvalues[0] - shift) * inv_scale,
                                      (eigenvalues[1] - shift) * inv_scale,
                                      (eigenvalues[2] - shift) * inv_scale};

  // Check if all eigenvalues are numerically the same
  constexpr T epsilon = std::numeric_limits<T>::epsilon();
  T eigen_range = scaled_eigenvalues[2] - scaled_eigenvalues[0];

  if (eigen_range <= epsilon) {
    // All three eigenvalues are numerically the same
    eigenvectors[0] = tf::vector<T, 3>{T(1), T(0), T(0)};
    eigenvectors[1] = tf::vector<T, 3>{T(0), T(1), T(0)};
    eigenvectors[2] = tf::vector<T, 3>{T(0), T(0), T(1)};
  } else {
    // Compute eigenvector of most distinct eigenvalue
    T d0 = scaled_eigenvalues[2] - scaled_eigenvalues[1];
    T d1 = scaled_eigenvalues[1] - scaled_eigenvalues[0];
    std::size_t k = 0, l = 2;
    if (d0 > d1) {
      std::swap(k, l);
      d0 = d1;
    }

    // Compute eigenvector of index k
    std::array<std::array<T, 3>, 3> tmp = scaled_mat;
    T eval_k = scaled_eigenvalues[k];
    tmp[0][0] -= eval_k;
    tmp[1][1] -= eval_k;
    tmp[2][2] -= eval_k;

    tf::vector<T, 3> representative;
    eigenvectors[k] = impl::extract_kernel_3x3(tmp, representative);
    eigenvectors[l] = representative;

    // Compute eigenvector of index l
    T threshold = T(2) * epsilon * d1;
    if (d0 <= threshold) {
      // Two other eigenvalues are numerically the same
      // Ortho-normalize the representative
      T projection = tf::dot(eigenvectors[k], eigenvectors[l]);
      eigenvectors[l] = eigenvectors[l] - projection * eigenvectors[k];
      eigenvectors[l] = tf::normalized(eigenvectors[l]);
    } else {
      tmp = scaled_mat;
      T eval_l = scaled_eigenvalues[l];
      tmp[0][0] -= eval_l;
      tmp[1][1] -= eval_l;
      tmp[2][2] -= eval_l;

      tf::vector<T, 3> dummy;
      eigenvectors[l] = impl::extract_kernel_3x3(tmp, dummy);
    }

    // Compute last eigenvector from the other two
    eigenvectors[1] =
        tf::normalized(tf::cross(eigenvectors[2], eigenvectors[0]));
  }

  return std::array<tf::unit_vector<T, 3>, 3>{
      tf::make_unit_vector(tf::unsafe, eigenvectors[0]),
      tf::make_unit_vector(tf::unsafe, eigenvectors[1]),
      tf::make_unit_vector(tf::unsafe, eigenvectors[2])};
}

} // namespace tf::core

namespace tf::core {
namespace impl {

// 2D analogue of extract_kernel_3x3: find a normalized vector in the kernel of
// a 2x2 matrix.
template <typename T>
auto extract_kernel_2x2(const std::array<std::array<T, 2>, 2> &mat)
    -> tf::vector<T, 2> {
  using std::abs;

  const T a = mat[0][0];
  const T b = mat[0][1];
  const T c = mat[1][0];
  const T d = mat[1][1];

  // Choose the row with the larger norm for better numerical stability.
  const T n0 = abs(a) + abs(b);
  const T n1 = abs(c) + abs(d);

  tf::vector<T, 2> v;

  if (n0 >= n1) {
    // Row 0: (a, b). Any vector perpendicular to it is an eigenvector
    // candidate.
    v = tf::vector<T, 2>{-b, a};
  } else {
    // Row 1: (c, d).
    v = tf::vector<T, 2>{-d, c};
  }

  T norm2 = tf::dot(v, v);

  if (norm2 <= std::numeric_limits<T>::epsilon()) {
    // Degenerate row, fall back to a fixed direction.
    v = tf::vector<T, 2>{T(1), T(0)};
    norm2 = T(1);
  }

  return v / tf::sqrt(norm2);
}

} // namespace impl

template <typename T>
auto eigen_vectors_of(const std::array<std::array<T, 2>, 2> &m,
                      const std::array<T, 2> &eigenvalues) {
  using std::abs;

  std::array<tf::vector<T, 2>, 2> eigenvectors;

  // Check if eigenvalues are numerically the same
  const T max_eval = std::max(abs(eigenvalues[0]), abs(eigenvalues[1]));
  const T diff = abs(eigenvalues[1] - eigenvalues[0]);
  const T tol = std::numeric_limits<T>::epsilon() * std::max(T(1), max_eval);

  if (diff <= tol) {
    // Nearly repeated eigenvalue: any orthonormal basis will do.
    eigenvectors[0] = tf::vector<T, 2>{T(1), T(0)};
    eigenvectors[1] = tf::vector<T, 2>{T(0), T(1)};
  } else {
    // Compute eigenvector for λ0
    {
      std::array<std::array<T, 2>, 2> A = m;
      A[0][0] -= eigenvalues[0];
      A[1][1] -= eigenvalues[0];
      eigenvectors[0] = impl::extract_kernel_2x2(A);
    }

    // Compute eigenvector for λ1
    {
      std::array<std::array<T, 2>, 2> A = m;
      A[0][0] -= eigenvalues[1];
      A[1][1] -= eigenvalues[1];
      eigenvectors[1] = impl::extract_kernel_2x2(A);
    }

    // Orthonormalize v1 against v0 to improve numerical robustness
    const T proj = tf::dot(eigenvectors[0], eigenvectors[1]);
    eigenvectors[1] = eigenvectors[1] - proj * eigenvectors[0];
    eigenvectors[1] = tf::normalized(eigenvectors[1]);
  }

  return std::array<tf::unit_vector<T, 2>, 2>{
      tf::make_unit_vector(tf::unsafe, eigenvectors[0]),
      tf::make_unit_vector(tf::unsafe, eigenvectors[1])};
}

} // namespace tf::core

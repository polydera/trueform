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

#include "../sqrt.hpp"
#include <cmath>
#include <cstddef>
#include <limits>

namespace tf::linalg {

/// @ingroup linalg
/// @brief Workspace size for solve_least_squares (in elements of type T).
///
/// @tparam T The scalar type.
/// @param rows Number of rows (equations).
/// @param cols Number of columns (unknowns).
/// @return Required workspace size in elements.
template <typename T>
constexpr auto least_squares_workspace_size(std::size_t rows,
                                            std::size_t cols) -> std::size_t {
  // b copy + norms_upd + norms_dir + perm (ceil to T-sized slots)
  std::size_t perm_slots =
      (cols * sizeof(std::size_t) + sizeof(T) - 1) / sizeof(T);
  return rows + 2 * cols + perm_slots;
}

/// @ingroup linalg
/// @brief Solve least squares min||Ax - b||₂ using column-pivoting Householder
/// QR.
///
/// Uses in-place column-pivoting Householder QR decomposition with stable norm
/// downdate (LAPACK lawn176). Optimized for tall-skinny matrices (rows >> cols).
///
/// Storage: A is column-major (Fortran order). For an N×M matrix:
///   A[i + j*rows] accesses element (i,j)
///
/// @tparam T The scalar type (float or double).
/// @param A Column-major N×M matrix, MODIFIED IN-PLACE (becomes R upper +
/// Householder lower).
/// @param b N×1 right-hand side (not modified).
/// @param x M×1 output solution.
/// @param rows N (number of equations).
/// @param cols M (number of unknowns).
/// @param work Workspace of size least_squares_workspace_size<T>(rows, cols).
/// @param thresh Rank threshold (negative = auto).
template <typename T>
auto solve_least_squares(T *A, const T *b, T *x, std::size_t rows,
                         std::size_t cols, T *work, T thresh = T(-1)) -> void {
  using std::abs;

  const std::size_t size = (rows < cols) ? rows : cols;

  // Workspace layout:
  // [0, rows)           : bw (b copy, becomes Q^T b)
  // [rows, rows+cols)   : norms_upd[cols]
  // [rows+cols, rows+2*cols) : norms_dir[cols]
  // [rows+2*cols, ...]  : perm[cols] (as std::size_t)
  T *bw = work;
  T *norms_upd = bw + rows;
  T *norms_dir = norms_upd + cols;
  std::size_t *perm = reinterpret_cast<std::size_t *>(norms_dir + cols);

  // Copy b (A is modified in-place)
  for (std::size_t i = 0; i < rows; ++i)
    bw[i] = b[i];

  // Initialize permutation and column norms
  T max_norm = T(0);
  for (std::size_t j = 0; j < cols; ++j) {
    perm[j] = j;
    T sq = T(0);
    const T *col = A + j * rows;
    for (std::size_t i = 0; i < rows; ++i)
      sq += col[i] * col[i];
    norms_upd[j] = norms_dir[j] = tf::sqrt(sq);
    if (norms_upd[j] > max_norm)
      max_norm = norms_upd[j];
  }

  // Auto threshold (matches Eigen: eps * max_norm, scaled by sqrt(size))
  if (thresh < T(0))
    thresh = std::numeric_limits<T>::epsilon() * max_norm * tf::sqrt(T(size));

  const T norm_recompute_thresh = tf::sqrt(std::numeric_limits<T>::epsilon());
  std::size_t rank = size;

  // Main factorization loop
  for (std::size_t k = 0; k < size; ++k) {
    // Column pivot: find largest norm among columns k:cols
    std::size_t pivot = k;
    T best = norms_upd[k];
    for (std::size_t j = k + 1; j < cols; ++j) {
      if (norms_upd[j] > best) {
        best = norms_upd[j];
        pivot = j;
      }
    }

    if (best < thresh) {
      rank = k;
      break;
    }

    // Swap columns k <-> pivot
    if (pivot != k) {
      T *ck = A + k * rows;
      T *cp = A + pivot * rows;
      for (std::size_t i = 0; i < rows; ++i) {
        T t = ck[i];
        ck[i] = cp[i];
        cp[i] = t;
      }
      T t = norms_upd[k];
      norms_upd[k] = norms_upd[pivot];
      norms_upd[pivot] = t;
      t = norms_dir[k];
      norms_dir[k] = norms_dir[pivot];
      norms_dir[pivot] = t;
      std::size_t ti = perm[k];
      perm[k] = perm[pivot];
      perm[pivot] = ti;
    }

    // Compute Householder reflector H = I - tau * v * v^T
    // v = [1, essential...], applied to column k rows k:N
    T *col_k = A + k * rows;
    T tail_sq = T(0);
    for (std::size_t i = k + 1; i < rows; ++i)
      tail_sq += col_k[i] * col_k[i];

    T x0 = col_k[k];
    T tau, beta;

    if (tail_sq <= std::numeric_limits<T>::min()) {
      tau = T(0);
      beta = x0;
    } else {
      beta = tf::sqrt(x0 * x0 + tail_sq);
      if (x0 >= T(0))
        beta = -beta;
      T denom = x0 - beta;
      for (std::size_t i = k + 1; i < rows; ++i)
        col_k[i] /= denom;
      tau = (beta - x0) / beta;
    }
    col_k[k] = beta; // R[k,k]

    if (tau != T(0)) {
      // Apply H to remaining columns
      for (std::size_t j = k + 1; j < cols; ++j) {
        T *cj = A + j * rows;
        T dot = cj[k];
        for (std::size_t i = k + 1; i < rows; ++i)
          dot += col_k[i] * cj[i];
        cj[k] -= tau * dot;
        for (std::size_t i = k + 1; i < rows; ++i)
          cj[i] -= tau * dot * col_k[i];
      }

      // Apply H to b (builds Q^T b incrementally)
      T dot = bw[k];
      for (std::size_t i = k + 1; i < rows; ++i)
        dot += col_k[i] * bw[i];
      bw[k] -= tau * dot;
      for (std::size_t i = k + 1; i < rows; ++i)
        bw[i] -= tau * dot * col_k[i];
    }

    // Stable norm downdate (LAPACK lawn176)
    for (std::size_t j = k + 1; j < cols; ++j) {
      if (norms_upd[j] != T(0)) {
        T r = abs(A[j * rows + k]) / norms_upd[j];
        T temp = (T(1) + r) * (T(1) - r);
        if (temp < T(0))
          temp = T(0);
        T ratio = norms_upd[j] / norms_dir[j];
        if (temp * ratio * ratio <= norm_recompute_thresh) {
          // Recompute directly
          T sq = T(0);
          const T *cj = A + j * rows;
          for (std::size_t i = k + 1; i < rows; ++i)
            sq += cj[i] * cj[i];
          norms_dir[j] = norms_upd[j] = tf::sqrt(sq);
        } else {
          norms_upd[j] *= tf::sqrt(temp);
        }
      }
    }
  }

  // Back-substitution: R * y = (Q^T b)[0:rank], then x[perm] = y
  for (std::size_t i = 0; i < cols; ++i)
    x[i] = T(0);

  for (std::size_t kk = rank; kk-- > 0;) {
    T sum = bw[kk];
    for (std::size_t j = kk + 1; j < rank; ++j)
      sum -= A[j * rows + kk] * x[perm[j]];
    x[perm[kk]] = sum / A[kk * rows + kk];
  }
}

} // namespace tf::linalg

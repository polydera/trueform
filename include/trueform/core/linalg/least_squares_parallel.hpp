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

#include "../algorithm/block_reduce.hpp"
#include "../buffer.hpp"
#include "../sqrt.hpp"
#include "../views/sequence_range.hpp"
#include "./least_squares.hpp"
#include <cmath>
#include <cstddef>
#include <limits>

namespace tf::linalg {

namespace detail {

/// @brief Non-pivoted Householder QR for TSQR local blocks.
///
/// Computes A = Q*R in-place (R in upper triangle, Householder vectors below).
/// Also computes Q^T * b in-place in b_inout.
///
/// Non-pivoted is required for TSQR so that R factors from different blocks
/// have consistent column ordering for stacking.
///
/// Uses the same Householder reflector computation as solve_least_squares
/// for numerical consistency.
///
/// @param A Column-major rows×cols matrix, modified in-place.
/// @param b_inout Input: b vector (rows×1). Output: Q^T b (rows×1).
/// @param rows Number of rows.
/// @param cols Number of columns.
/// @return The effective rank (number of non-negligible diagonal elements).
template <typename T>
auto qr_factorize_inplace(T *A, T *b_inout, std::size_t rows,
                          std::size_t cols) -> std::size_t {
  using std::abs;

  const std::size_t size = (rows < cols) ? rows : cols;

  // Compute initial column norms for rank threshold
  T max_norm = T(0);
  for (std::size_t j = 0; j < cols; ++j) {
    T sq = T(0);
    const T *col = A + j * rows;
    for (std::size_t i = 0; i < rows; ++i)
      sq += col[i] * col[i];
    T norm = tf::sqrt(sq);
    if (norm > max_norm)
      max_norm = norm;
  }

  const T thresh =
      std::numeric_limits<T>::epsilon() * max_norm * tf::sqrt(T(size));
  std::size_t rank = size;

  for (std::size_t k = 0; k < size; ++k) {
    T *col_k = A + k * rows;

    // Compute Householder reflector H = I - tau * v * v^T
    // v = [1, essential...], applied to column k rows k:rows
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

    // Check for rank deficiency
    if (abs(beta) < thresh) {
      rank = k;
      break;
    }

    if (tau != T(0)) {
      // Apply H to remaining columns of A
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
      T dot = b_inout[k];
      for (std::size_t i = k + 1; i < rows; ++i)
        dot += col_k[i] * b_inout[i];
      b_inout[k] -= tau * dot;
      for (std::size_t i = k + 1; i < rows; ++i)
        b_inout[i] -= tau * dot * col_k[i];
    }
  }

  return rank;
}

/// @brief Extract upper triangular R from factorized matrix.
///
/// @param A Factorized matrix (rows×cols, column-major).
/// @param R Output R matrix (cols×cols, column-major).
/// @param rows Number of rows in A.
/// @param cols Number of columns.
/// @param rank Effective rank (zeros out rows beyond rank).
template <typename T>
auto extract_R(const T *A, T *R, std::size_t rows, std::size_t cols,
               std::size_t rank) -> void {
  // R is stored in upper triangle of A
  // Copy to cols×cols dense matrix
  for (std::size_t j = 0; j < cols; ++j) {
    for (std::size_t i = 0; i < cols; ++i) {
      if (i <= j && i < rank)
        R[i + j * cols] = A[i + j * rows];
      else
        R[i + j * cols] = T(0);
    }
  }
}

} // namespace detail

/// @ingroup linalg
/// @brief Workspace state for parallel TSQR least squares.
///
/// Reusable across calls to avoid repeated allocations.
template <typename T> struct parallel_least_squares_state {
  tf::buffer<T> R_stacked;   ///< Stacked R factors from all blocks
  tf::buffer<T> Qtb_stacked; ///< Stacked Q^T b vectors from all blocks
  tf::buffer<T> final_work;  ///< Workspace for final solve
};

/// @ingroup linalg
/// @brief Local workspace for each TSQR block (thread-local).
template <typename T> struct tsqr_local_state {
  tf::buffer<T> block_A; ///< Copy of block for in-place QR
  tf::buffer<T> block_b; ///< Copy of block's b for in-place Q^T b
  tf::buffer<T> R;       ///< Extracted R factor (cols×cols)
  tf::buffer<T> Qtb;     ///< First cols elements of Q^T b
};

/// @ingroup linalg
/// @brief Solve least squares min||Ax - b||₂ using parallel TSQR.
///
/// Uses Tall Skinny QR (TSQR) algorithm:
/// 1. Divide matrix into row-blocks
/// 2. Compute local QR on each block in parallel
/// 3. Stack R factors and solve final (small) system
///
/// Numerically stable (same as sequential QR) and parallelizes well
/// for tall-skinny matrices where rows >> cols.
///
/// @tparam T The scalar type (float or double).
/// @param A Column-major N×M matrix (not modified).
/// @param b N×1 right-hand side (not modified).
/// @param x M×1 output solution.
/// @param rows N (number of equations).
/// @param cols M (number of unknowns).
/// @param state Reusable workspace.
template <typename T>
auto solve_least_squares_parallel(const T *A, const T *b, T *x, std::size_t rows,
                                  std::size_t cols,
                                  parallel_least_squares_state<T> &state)
    -> void {
  // For small problems, use sequential solver (parallel overhead not worth it)
  if (rows < cols * 20 || rows < 15000) {
    tf::buffer<T> A_copy;
    A_copy.allocate(rows * cols);
    std::copy(A, A + rows * cols, A_copy.data());

    tf::buffer<T> work;
    work.allocate(least_squares_workspace_size<T>(rows, cols));
    solve_least_squares(A_copy.data(), b, x, rows, cols, work.data());
    return;
  }

  // Clear stacked buffers
  state.R_stacked.clear();
  state.Qtb_stacked.clear();

  // Use blocked_reduce: each block computes local QR, aggregates R and Qtb
  tf::blocked_reduce(
      tf::make_sequence_range(rows),
      // Global result: tuple of references to stacked buffers
      std::forward_as_tuple(state.R_stacked, state.Qtb_stacked),
      // Local result template
      tsqr_local_state<T>{},
      // Task: process a block of rows
      [A, b, rows, cols](const auto &range, tsqr_local_state<T> &local) {
        const std::size_t start = *range.begin();
        const std::size_t end = *(range.end() - 1) + 1;
        const std::size_t block_rows = end - start;

        if (block_rows < cols) {
          // Block too small for full-rank QR, skip
          // (will be handled by other blocks)
          local.R.clear();
          local.Qtb.clear();
          return;
        }

        // Allocate local workspace
        local.block_A.allocate(block_rows * cols);
        local.block_b.allocate(block_rows);
        local.R.allocate(cols * cols);
        local.Qtb.allocate(cols);

        // Copy block of A (column-major)
        for (std::size_t j = 0; j < cols; ++j) {
          for (std::size_t i = 0; i < block_rows; ++i) {
            local.block_A[i + j * block_rows] = A[start + i + j * rows];
          }
        }

        // Copy block of b
        for (std::size_t i = 0; i < block_rows; ++i) {
          local.block_b[i] = b[start + i];
        }

        // Compute local QR in-place, get effective rank
        std::size_t rank = detail::qr_factorize_inplace(
            local.block_A.data(), local.block_b.data(), block_rows, cols);

        if (rank == 0) {
          // Completely degenerate block, skip
          local.R.clear();
          local.Qtb.clear();
          return;
        }

        // Extract R (upper cols×cols of factorized A)
        detail::extract_R(local.block_A.data(), local.R.data(), block_rows,
                          cols, rank);

        // Extract Q^T b (first cols elements)
        for (std::size_t i = 0; i < cols; ++i) {
          local.Qtb[i] = local.block_b[i];
        }
      },
      // Aggregate: stack R and Qtb into global buffers
      [cols](const tsqr_local_state<T> &local, auto &result) {
        auto &[R_stacked, Qtb_stacked] = result;

        if (local.R.size() == 0)
          return; // Skip empty blocks

        // Append R (cols×cols, column-major)
        std::size_t R_old_size = R_stacked.size();
        R_stacked.reallocate(R_old_size + cols * cols);
        std::copy(local.R.begin(), local.R.end(),
                  R_stacked.begin() + R_old_size);

        // Append Qtb (cols×1)
        std::size_t Qtb_old_size = Qtb_stacked.size();
        Qtb_stacked.reallocate(Qtb_old_size + cols);
        std::copy(local.Qtb.begin(), local.Qtb.end(),
                  Qtb_stacked.begin() + Qtb_old_size);
      });

  // Now we have stacked R factors and Qtb vectors
  // The stacked system is (num_blocks * cols) × cols
  const std::size_t num_blocks = state.Qtb_stacked.size() / cols;
  const std::size_t stacked_rows = num_blocks * cols;

  if (num_blocks == 0) {
    // No valid blocks (shouldn't happen for reasonable input)
    for (std::size_t i = 0; i < cols; ++i)
      x[i] = T(0);
    return;
  }

  if (num_blocks == 1) {
    // Only one block: back-substitute directly
    // R is upper triangular, solve R*x = Qtb
    for (std::size_t kk = cols; kk-- > 0;) {
      T sum = state.Qtb_stacked[kk];
      for (std::size_t j = kk + 1; j < cols; ++j) {
        sum -= state.R_stacked[kk + j * cols] * x[j];
      }
      T diag = state.R_stacked[kk + kk * cols];
      x[kk] = (std::abs(diag) > std::numeric_limits<T>::min()) ? sum / diag
                                                               : T(0);
    }
    return;
  }

  // Multiple blocks: solve stacked system with pivoted QR
  // The stacked R factors form a (stacked_rows × cols) matrix
  // Use the robust sequential solver for this small system
  state.final_work.allocate(least_squares_workspace_size<T>(stacked_rows, cols));

  // R_stacked is stored as concatenated cols×cols blocks (column-major each)
  // Need to convert to single column-major stacked_rows×cols matrix
  tf::buffer<T> stacked_A;
  stacked_A.allocate(stacked_rows * cols);

  for (std::size_t block = 0; block < num_blocks; ++block) {
    const T *R_block = state.R_stacked.data() + block * cols * cols;
    const std::size_t row_offset = block * cols;
    for (std::size_t j = 0; j < cols; ++j) {
      for (std::size_t i = 0; i < cols; ++i) {
        stacked_A[row_offset + i + j * stacked_rows] = R_block[i + j * cols];
      }
    }
  }

  solve_least_squares(stacked_A.data(), state.Qtb_stacked.data(), x,
                      stacked_rows, cols, state.final_work.data());
}

/// @ingroup linalg
/// @brief Solve least squares using parallel TSQR (allocates internally).
///
/// Convenience overload that allocates workspace internally.
/// For repeated calls, prefer the overload with explicit state.
template <typename T>
auto solve_least_squares_parallel(const T *A, const T *b, T *x, std::size_t rows,
                                  std::size_t cols) -> void {
  parallel_least_squares_state<T> state;
  solve_least_squares_parallel(A, b, x, rows, cols, state);
}

} // namespace tf::linalg

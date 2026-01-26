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
#include "./eigen_values_of.hpp"
#include "./eigen_vectors_of.hpp"

namespace tf {

/// @ingroup core_algorithms
/// @brief Compute eigenvalues and eigenvectors of a 3x3 symmetric matrix.
///
/// Uses analytical closed-form solution for efficiency.
/// Eigenvalues are returned in ascending order.
///
/// @tparam T The scalar type.
/// @param mat The symmetric 3x3 matrix.
/// @return A pair of (eigenvalues array, eigenvectors array).
template <typename T>
auto eigen_of_symmetric(const std::array<std::array<T, 3>, 3> &mat) {
  auto eigenvalues = core::eigen_values_of(mat);
  auto eigenvectors = core::eigen_vectors_of(mat, eigenvalues);
  return std::make_pair(eigenvalues, eigenvectors);
}

/// @ingroup core_algorithms
/// @brief Compute eigenvalues and eigenvectors of a 2x2 symmetric matrix.
/// @overload
template <typename T>
auto eigen_of_symmetric(const std::array<std::array<T, 2>, 2> &mat) {
  auto eigenvalues = core::eigen_values_of(mat);
  auto eigenvectors = core::eigen_vectors_of(mat, eigenvalues);
  return std::make_pair(eigenvalues, eigenvectors);
}
} // namespace tf

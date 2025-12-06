/*
 * Copyright (c) 2025 Žiga Sajovic, XLAB
 * Licensed for noncommercial use under the PolyForm Noncommercial
 * License 1.0.0. Commercial licensing available via ziga.sajovic@xlab.si.
 * https://github.com/xlabmedical/trueform
 */
#pragma once
#include "./eigen_values_of.hpp"
#include "./eigen_vectors_of.hpp"

namespace tf {

template <typename T>
auto eigen_of_symmetric(const std::array<std::array<T, 3>, 3> &mat) {
  auto eigenvalues = core::eigen_values_of(mat);
  auto eigenvectors = core::eigen_vectors_of(mat, eigenvalues);
  return std::make_pair(eigenvalues, eigenvectors);
}

template <typename T>
auto eigen_of_symmetric(const std::array<std::array<T, 2>, 2> &mat) {
  auto eigenvalues = core::eigen_values_of(mat);
  auto eigenvectors = core::eigen_vectors_of(mat, eigenvalues);
  return std::make_pair(eigenvalues, eigenvectors);
}
} // namespace tf

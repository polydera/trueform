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

namespace tf::linalg {
template <typename MatrixTypeIn, typename MatrixTypeOut>
void invert_matrix_2x2(const MatrixTypeIn &in_matrix,
                       MatrixTypeOut &&out_matrix) {
  // Extract individual elements for readability
  double a = in_matrix(0, 0), b = in_matrix(0, 1);
  double c = in_matrix(1, 0), d = in_matrix(1, 1);

  // Compute the determinant
  double det = a * d - b * c;

  // Compute the inverse matrix elements
  out_matrix(0, 0) = d / det;  // C11
  out_matrix(0, 1) = -b / det; // C12
  out_matrix(1, 0) = -c / det; // C21
  out_matrix(1, 1) = a / det;  // C22
}
} // namespace tf::linalg

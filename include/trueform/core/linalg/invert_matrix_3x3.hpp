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
auto invert_matrix_3x3(const MatrixTypeIn &in_matrix,
                       MatrixTypeOut &&out_matrix) {
  double a = in_matrix(0, 0), b = in_matrix(0, 1), c = in_matrix(0, 2);
  double d = in_matrix(1, 0), e = in_matrix(1, 1), f = in_matrix(1, 2);
  double g = in_matrix(2, 0), h = in_matrix(2, 1), i = in_matrix(2, 2);

  // Compute the determinant
  double det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);

  // Compute the adjugate matrix (transpose of the cofactor matrix)
  out_matrix(0, 0) = (e * i - f * h) / det;  // C11
  out_matrix(0, 1) = -(b * i - c * h) / det; // C12
  out_matrix(0, 2) = (b * f - c * e) / det;  // C13
  out_matrix(1, 0) = -(d * i - f * g) / det; // C21
  out_matrix(1, 1) = (a * i - c * g) / det;  // C22
  out_matrix(1, 2) = -(a * f - c * d) / det; // C23
  out_matrix(2, 0) = (d * h - e * g) / det;  // C31
  out_matrix(2, 1) = -(a * h - b * g) / det; // C32
  out_matrix(2, 2) = (a * e - b * d) / det;  // C33
}
} // namespace tf::linalg

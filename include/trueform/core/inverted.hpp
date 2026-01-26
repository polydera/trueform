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
#include "./linalg/invert_matrix_2x2.hpp"
#include "./linalg/invert_matrix_3x3.hpp"
#include "./linalg/trans.hpp"
#include "./transformation_like.hpp"

namespace tf {
/// @ingroup core_primitives
/// @brief Computes the inverse of @tf::transformation
template <std::size_t Dims, typename Policy>
auto inverted(const transformation_like<Dims, Policy> &transform) {
  static_assert(Dims == 2 || Dims == 3);
  tf::transformation_like<Dims,
                          tf::linalg::trans<typename Policy::value_type, Dims>>
      out;
  if constexpr (Dims == 3)
    linalg::invert_matrix_3x3(transform, out);
  else
    linalg::invert_matrix_2x2(transform, out);
  for (std::size_t i = 0; i < Dims; ++i) {
    out(i, Dims) = 0;
    if constexpr (Policy::n_columns == Dims + 1)
      for (std::size_t j = 0; j < Dims; ++j) {
        out(i, Dims) -= out(i, j) * transform(j, Dims);
      }
  }
  return out;
}

} // namespace tf


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

#include "trueform/core/buffer.hpp"
#include "trueform/core/transformation_like.hpp"
#include "trueform/ts/core/wasm_ndarray.hpp"

namespace tf {
namespace ts {

/// Convert a transformation_like<Dims, Policy> to a wasm_ndarray [(Dims+1),
/// (Dims+1)] of element type Real.
///
/// tf::transformation stores only Dims × (Dims+1) values (e.g. 3×4 = 12 for
/// 3D). The implicit last row [0, ..., 0, 1] is filled in.
template <typename Real, std::size_t Dims, typename Policy>
auto to_wasm_matrix(const tf::transformation_like<Dims, Policy> &t)
    -> wasm_ndarray<Real> {
  constexpr std::size_t N = Dims + 1;
  tf::buffer<Real> buf;
  buf.allocate(N * N);
  auto *ptr = buf.data();
  for (std::size_t i = 0; i < Dims; ++i)
    for (std::size_t j = 0; j < N; ++j)
      ptr[i * N + j] = static_cast<Real>(t(i, j));
  for (std::size_t j = 0; j < Dims; ++j)
    ptr[Dims * N + j] = Real{0};
  ptr[Dims * N + Dims] = Real{1};
  return wasm_ndarray<Real>::from_buffer(std::move(buf),
                                         {static_cast<int>(N),
                                          static_cast<int>(N)});
}

} // namespace ts
} // namespace tf

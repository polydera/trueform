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

#include "trueform/cpp/core/matrix.hpp"
#include "trueform/cpp/core/mesh.hpp"

#include <cstddef>
#include <cstdint>

namespace tf::cpp {
/// @brief V - E + F, each undirected edge counted once.
/// @note Reads the MANIFOLD EDGE LINK.
template <typename Index, typename Real, std::size_t Dims, std::size_t Ngon>
auto euler_characteristic(const mesh<Index, Real, Dims, Ngon> &value)
    -> std::int32_t;

#define TF_CPP_EXTERN_EULER_CHARACTERISTIC(Index, Real, Dims, Ngon)            \
  extern template auto euler_characteristic<Index, Real, Dims, Ngon>(          \
      const mesh<Index, Real, Dims, Ngon> &) -> std::int32_t

TF_CPP_MATRIX_FOR_EACH_INDEX_REAL_DIMS_NGON(TF_CPP_EXTERN_EULER_CHARACTERISTIC)

#undef TF_CPP_EXTERN_EULER_CHARACTERISTIC

} // namespace tf::cpp
